"""
config_and_dataset_v2.py — BUSI Dataset Configuration and Loader
Handles the standard BUSI layout from:
  Al-Dhabyani W et al. "Dataset of breast ultrasound images." Data in Brief, 2020.

Revision log (v2 → v3 — five targeted fixes):
─────────────────────────────────────────────────────────────────────────────
Fix 1  [Vanilla KD imbalance]
    cfg.temperature reduced from 4.0 → 2.0.  The T² loss scaling in
    VanillaKDLoss (removed in models_and_kd_v2.py) was compounding with T=4
    to inflate the KD term by 16×.  At T=2 the soft-label effect is meaningful
    while T²=4 leaves λ_kd as the sole balance knob.

Fix 2  [Multi-seed support]
    cfg.seeds  (List[int], default [42, 123, 456]) specifies which random seeds
    to train over.  train.py's --seed flag controls the active seed per run;
    benchmark_only mode aggregates all per-seed checkpoints automatically.

Fix 3  [Teacher test metrics]
    No config change required; handled entirely in train.py.

Fix 4  [LDL hyperparameter defaults for ResNet50 + MobileNetV2]
    ldl_embed_dim   : 128 → 256   (ResNet50 feat_idx=-2 outputs 1024ch;
                                   D=256 gives a richer projection target)
    ldl_num_anchors : 32  → 64    (more anchors → finer Laguerre partition
                                   covering BUSI lesion boundaries better)

Fix 5  [λ_LDL ramp-up]
    ldl_lambda_ramp_epochs : int = 10   (new field)
    The effective KD weight ramps from 0 → λ_kd linearly over the first N
    epochs after the ψ_T warm-up phase.  This prevents the abrupt loss spike
    observed at epoch 25 in the v2 run.

Architecture change:
    Teacher : ResNet50-UNet    (~32M params)  — was ResNet34
    Student : MobileNetV2-UNet (~3.5M params) — Reverted back from ShuffleNetV2
                                                due to timm missing the model.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ================================================================
# CONFIGURATION
# ================================================================
@dataclass
class Config:
    # --- Paths ---
    data_root:   str = "./BUSI"
    ckpt_dir:    str = "./checkpoints"
    results_dir: str = "./results"
    log_dir:     str = "./logs"

    # --- Data ---
    img_size:    int   = 256
    batch_size:  int   = 8
    num_workers: int   = 2
    train_split: float = 0.70
    val_split:   float = 0.15      # test = 1 − train − val
    seed:        int   = 42        # active seed for one run
    use_classes: tuple = ("benign", "malignant")

    # Fix 2: multi-seed list.  Each value is passed as --seed to a separate
    # train.py invocation.  benchmark_only mode aggregates all seeds.
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    # Normalisation statistics — None → computed from training images at runtime.
    img_mean: Optional[Tuple[float, float, float]] = None
    img_std:  Optional[Tuple[float, float, float]] = None

    # --- Training ---
    epochs:       int   = 100
    lr:           float = 1e-4
    weight_decay: float = 1e-5
    lr_patience:  int   = 15       # ReduceLROnPlateau patience
    grad_clip:    float = 1.0
    amp:          bool  = True
    early_stop:   int   = 25

    # --- KD global ---
    lambda_kd:   float = 0.5

    # Fix 1: temperature reduced from 4.0 → 2.0.
    # VanillaKDLoss no longer applies T² scaling (removed in models_and_kd_v2.py).
    # At T=2 the soft-label smoothing is still meaningful; T²=4 is benign.
    temperature: float = 2.0

    # --- LDL hyperparameters ---
    # Fix 4: embed_dim 128 → 256 (ResNet50 distil feats are 1024ch; D=256 richer)
    ldl_embed_dim:   int   = 256
    # Fix 4: num_anchors 32 → 64 (finer Laguerre partition for boundary coverage)
    ldl_num_anchors: int   = 64
    ldl_alpha:       float = 0.5    # spatial vs. semantic cost balance
    ldl_k:           float = 1.0    # transport-cost scale
    ldl_a1:          float = 0.30   # TV over-supply penalty
    ldl_b1:          float = 0.10   # TV under-supply penalty
    ldl_a2:          float = 1.00   # TV over-demand penalty
    ldl_b2:          float = 0.20   # TV under-demand penalty
    ldl_T_w:         int   = 15     # dual-solver inner steps per batch
    ldl_eta0:        float = 0.05   # dual-solver initial step size
    ldl_beta:        float = 0.60   # dual-solver step-size decay exponent
    ldl_anc_reg:     float = 0.01   # anchor regularisation weight γ
    ldl_anc_reg_tau: float = 0.10   # soft-min temperature τ (Issue 4 fix)

    # Fix 5: λ_LDL ramp-up.  The effective KD weight scales linearly from 0
    # to lambda_kd over this many epochs AFTER the ψ_T warm-up completes.
    # Set to 0 to use a hard switch (v2 behaviour, causes a loss spike).
    ldl_lambda_ramp_epochs: int = 10

    # --- ψ_T warm-up (Issue 5 — k-means++ runs AFTER this many steps) ---
    psi_T_warmup_steps: int = 500

    # --- Architectures ---
    # Teacher: ResNet50-UNet, ~32 M params — stronger backbone than ResNet34.
    teacher_encoder: str = "resnet50"
    # Student: MobileNetV2-UNet, ~3.5 M params.
    #   Reverted from ShuffleNetV2 (tu-shufflenet_v2_x1_0) because timm does not
    #   have a shufflenet_v2_x1_0 model, leading to a RuntimeError.
    student_encoder: str = "mobilenet_v2"
    encoder_weights: str = "imagenet"
    # Penultimate encoder block used for distillation.
    # ResNet50    feat_idx=-2 → Layer3 → 1024 channels
    # MobileNetV2 feat_idx=-2 →        →   96 channels (auto-detected at runtime)
    distill_feat_idx: int = -2

    # --- Efficiency benchmark ---
    benchmark_repeats: int = 200
    benchmark_warmup:  int = 20

    # --- Device ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        for d in (self.ckpt_dir, self.results_dir, self.log_dir):
            os.makedirs(d, exist_ok=True)


# ================================================================
# DATASET STATISTICS (single-pass Welford accumulation)
# ================================================================
def compute_dataset_stats(
    data_root:   str,
    img_size:    int,
    classes:     Tuple[str, ...] = ("benign", "malignant"),
    max_samples: int = 500,
    seed:        int = 42,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute per-channel pixel mean and std from a random sample of images.

    Uses single-pass accumulation of Σx and Σx² (O(1) memory regardless of
    dataset size).  Falls back to ImageNet defaults if no images are found.
    """
    root = Path(data_root)
    candidate = root / "Dataset_BUSI_with_GT"
    if candidate.exists():
        root = candidate

    all_imgs: List[str] = []
    for cls in classes:
        cls_dir = root / cls
        if not cls_dir.exists():
            continue
        for p in cls_dir.iterdir():
            if (p.suffix.lower() in (".png", ".jpg", ".bmp")
                    and "_mask" not in p.name):
                all_imgs.append(str(p))

    if not all_imgs:
        print("  [Stats] No images found — falling back to ImageNet defaults.")
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    rng    = random.Random(seed)
    sample = (all_imgs if len(all_imgs) <= max_samples
              else rng.sample(all_imgs, max_samples))

    sum_  = np.zeros(3, dtype=np.float64)
    sum2_ = np.zeros(3, dtype=np.float64)
    n_pix = 0

    for path in sample:
        img = np.array(
            Image.open(path).convert("RGB").resize(
                (img_size, img_size), Image.BILINEAR),
            dtype=np.float64
        ) / 255.0
        pixels  = img.reshape(-1, 3)
        sum_   += pixels.sum(axis=0)
        sum2_  += (pixels ** 2).sum(axis=0)
        n_pix  += pixels.shape[0]

    mean = sum_ / n_pix
    var  = sum2_ / n_pix - mean ** 2
    std  = np.sqrt(np.maximum(var, 1e-8))

    mean_t = tuple(float(v) for v in mean)
    std_t  = tuple(float(v) for v in std)
    print(
        f"  [Stats] Computed from {len(sample)} images → "
        f"mean={tuple(round(v, 4) for v in mean_t)}  "
        f"std={tuple(round(v, 4) for v in std_t)}"
    )
    return mean_t, std_t


# ================================================================
# AUGMENTATION PIPELINES
# ================================================================
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(
    img_size: int,
    mean: Tuple[float, float, float] = _IMAGENET_MEAN,
    std:  Tuple[float, float, float] = _IMAGENET_STD,
) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        # CLAHE: local contrast enhancement — critical for US boundary clarity
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        # Spatial augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15,
                           rotate_limit=30, p=0.5, border_mode=0),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05,
                           alpha_affine=120 * 0.03, p=0.3),
        # Intensity augmentations
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.4),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(
    img_size: int,
    mean: Tuple[float, float, float] = _IMAGENET_MEAN,
    std:  Tuple[float, float, float] = _IMAGENET_STD,
) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# ================================================================
# BUSI DATASET
# ================================================================
class BUSIDataset(Dataset):
    """
    BUSI (Breast Ultrasound Images) Dataset.

    Supported directory layouts:
      Variant A:  data_root/benign/benign (1).png + benign (1)_mask.png
      Variant B:  data_root/Dataset_BUSI_with_GT/benign/...

    Multi-mask support
    ------------------
    Images with multiple annotated lesions have sibling mask files:
        benign (100)_mask.png, benign (100)_mask_1.png, ...
    All sibling masks are merged with pixel-wise maximum (logical OR) so that
    every annotated lesion contributes to the training signal.
    """

    def __init__(
        self,
        data_root:   str,
        split:       str   = "train",
        transforms:  Optional[A.Compose] = None,
        classes:     Tuple[str, ...] = ("benign", "malignant"),
        train_ratio: float = 0.70,
        val_ratio:   float = 0.15,
        seed:        int   = 42,
    ):
        self.transforms = transforms
        self.samples: List[Tuple[str, List[str]]] = []

        root = Path(data_root)
        candidate = root / "Dataset_BUSI_with_GT"
        if candidate.exists():
            root = candidate

        all_pairs: List[Tuple[str, List[str]]] = []

        for cls in classes:
            cls_dir = root / cls
            if not cls_dir.exists():
                print(f"  [WARN] class directory not found: {cls_dir}")
                continue

            imgs = sorted([
                p for p in cls_dir.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".bmp")
                and "_mask" not in p.name
            ])

            for img_path in imgs:
                mask_paths = _find_all_masks(img_path)
                if mask_paths:
                    all_pairs.append((str(img_path),
                                      [str(m) for m in mask_paths]))
                else:
                    print(f"  [WARN] mask not found for {img_path.name}")

        if len(all_pairs) == 0:
            raise FileNotFoundError(
                f"No image-mask pairs found in {root}. "
                "Check that BUSI data is extracted correctly."
            )

        rng = random.Random(seed)
        rng.shuffle(all_pairs)
        n       = len(all_pairs)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        if split == "train":
            self.samples = all_pairs[:n_train]
        elif split == "val":
            self.samples = all_pairs[n_train: n_train + n_val]
        elif split == "test":
            self.samples = all_pairs[n_train + n_val:]
        else:
            raise ValueError(f"split must be train/val/test, got {split!r}")

        multi_in_split = sum(1 for _, ms in self.samples if len(ms) > 1)
        print(
            f"  BUSIDataset [{split:5s}]: {len(self.samples)} samples "
            f"({multi_in_split} with merged multi-masks)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_paths = self.samples[idx]

        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        mask = np.zeros(
            np.array(Image.open(mask_paths[0]).convert("L"),
                     dtype=np.uint8).shape,
            dtype=np.uint8,
        )
        for mp in mask_paths:
            m    = np.array(Image.open(mp).convert("L"), dtype=np.uint8)
            mask = np.maximum(mask, m)

        mask = (mask > 127).astype(np.uint8)

        if self.transforms:
            aug  = self.transforms(image=img, mask=mask)
            img  = aug["image"]
            mask = aug["mask"].unsqueeze(0).float()
        else:
            img  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask[None]).float()

        return img, mask


# ================================================================
# PRIVATE HELPER — find all mask files for one image
# ================================================================
def _find_all_masks(img_path: Path) -> List[Path]:
    stem   = img_path.stem
    suffix = img_path.suffix
    found: List[Path] = []

    primary = img_path.with_name(stem + "_mask" + suffix)
    if primary.exists():
        found.append(primary)

    for k in range(1, 10):
        extra = img_path.with_name(stem + f"_mask_{k}" + suffix)
        if extra.exists():
            found.append(extra)
        else:
            break

    return found


# ================================================================
# DATALOADER BUILDER
# ================================================================
def build_dataloaders(cfg: Config):
    """
    Return (train_loader, val_loader, test_loader).

    Uses cfg.seed for the dataset split (ensures reproducible splits across
    seeds — all seeds see the same train/val/test partition, only model
    initialisation and data-augmentation order differ between seeds).

    If cfg.img_mean / cfg.img_std are None, per-channel statistics are
    computed automatically and stored back into cfg.
    """
    if cfg.img_mean is None or cfg.img_std is None:
        print("  Computing dataset normalisation statistics …")
        mean, std = compute_dataset_stats(
            data_root   = cfg.data_root,
            img_size    = cfg.img_size,
            classes     = cfg.use_classes,
            max_samples = 500,
            seed        = 42,   # fixed: stats derived from same image subset
        )
        cfg.img_mean = mean
        cfg.img_std  = std
    else:
        mean, std = cfg.img_mean, cfg.img_std

    train_tf = get_train_transforms(cfg.img_size, mean, std)
    val_tf   = get_val_transforms(cfg.img_size, mean, std)

    # NOTE: dataset split uses a fixed seed (42) so the test set is identical
    # across all training seeds.  Only weight initialisation varies.
    split_seed = 42

    train_ds = BUSIDataset(cfg.data_root, "train", train_tf,
                           cfg.use_classes, cfg.train_split,
                           cfg.val_split, split_seed)
    val_ds   = BUSIDataset(cfg.data_root, "val",   val_tf,
                           cfg.use_classes, cfg.train_split,
                           cfg.val_split, split_seed)
    test_ds  = BUSIDataset(cfg.data_root, "test",  val_tf,
                           cfg.use_classes, cfg.train_split,
                           cfg.val_split, split_seed)

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size  = cfg.batch_size,
            shuffle     = shuffle,
            num_workers = cfg.num_workers,
            pin_memory  = True,
            drop_last   = False,
        )

    return (make_loader(train_ds, True),
            make_loader(val_ds,   False),
            make_loader(test_ds,  False))