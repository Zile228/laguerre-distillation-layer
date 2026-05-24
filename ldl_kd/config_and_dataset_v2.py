"""
config.py + dataset.py — BUSI Dataset Configuration and Loader
Handles the standard BUSI layout from:
  Al-Dhabyani W et al. "Dataset of breast ultrasound images." Data in Brief, 2020.
Kaggle

Changes (v2 → v3):
  • Multi-mask merging: images with extra lesion masks (e.g. benign (1)_mask_1.png)
    are now handled correctly — all sibling masks are OR-merged at load time.
  • Dataset statistics: compute_dataset_stats() computes per-channel mean/std from
    the actual training images in a single pass.  build_dataloaders() calls it
    automatically when cfg.img_mean / cfg.img_std are None.
  • get_train_transforms / get_val_transforms now accept mean/std parameters so
    the normalisation uses the computed (or user-supplied) statistics.
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
    data_root:    str = "./BUSI"
    ckpt_dir:     str = "./checkpoints"
    results_dir:  str = "./results"
    log_dir:      str = "./logs"

    # --- Data ---
    img_size:     int   = 256
    batch_size:   int   = 8
    num_workers:  int   = 2
    train_split:  float = 0.70
    val_split:    float = 0.15   # test = 1 - train - val
    seed:         int   = 42
    use_classes:  tuple = ("benign", "malignant")  # exclude "normal" (no lesion)

    # Normalisation statistics.
    # Set to None to compute automatically from the training set at runtime
    # (recommended for medical images which differ from ImageNet statistics).
    # Set to explicit tuples to skip computation:
    #   img_mean = (0.485, 0.456, 0.406)
    #   img_std  = (0.229, 0.224, 0.225)
    img_mean: Optional[Tuple[float, float, float]] = None
    img_std:  Optional[Tuple[float, float, float]] = None

    # --- Training ---
    epochs:       int   = 100
    lr:           float = 1e-4
    weight_decay: float = 1e-5
    lr_patience:  int   = 15    # ReduceLROnPlateau patience
    grad_clip:    float = 1.0
    amp:          bool  = True  # mixed-precision (T4/P100 both support it)
    early_stop:   int   = 25    # stop if val Dice doesn't improve

    # --- KD global ---
    lambda_kd:    float = 0.5   # weight on KD loss  (Lseg + λ·Lkd)
    temperature:  float = 4.0   # vanilla KD temperature

    # --- LDL hyperparameters (Section 4.1 of paper) ---
    ldl_embed_dim:   int   = 128
    ldl_num_anchors: int   = 32
    ldl_alpha:       float = 0.5   # spatial vs semantic cost balance
    ldl_k:           float = 1.0   # transport cost scale
    ldl_a1:          float = 0.30  # TV over-supply penalty
    ldl_b1:          float = 0.10  # TV under-supply penalty
    ldl_a2:          float = 1.00  # TV over-demand penalty
    ldl_b2:          float = 0.20  # TV under-demand penalty
    ldl_T_w:         int   = 15    # dual solver steps per batch
    ldl_eta0:        float = 0.05  # dual solver initial step size
    ldl_beta:        float = 0.60  # dual solver step size decay exponent
    ldl_anc_reg:     float = 0.01  # anchor regularisation weight γ
    ldl_anc_reg_tau: float = 0.10  # soft-min temperature τ (Issue 4 fix)

    # --- ψ_T warm-up (Issue 5 fix: k-means++ runs AFTER this many steps) ---
    # Set to 0 to skip warm-up (not recommended).
    psi_T_warmup_steps: int = 500  # gradient steps before ψ_T is frozen

    # --- Architectures ---
    teacher_encoder:  str = "resnet34"      # ~24M params total
    student_encoder:  str = "mobilenet_v2"  # ~4M params total
    encoder_weights:  str = "imagenet"
    # Feature level used for distillation (-2 = penultimate encoder block)
    distill_feat_idx: int = -2

    # --- Efficiency benchmark ---
    benchmark_repeats: int = 200  # forward-pass repetitions for latency
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
    Compute per-channel pixel mean and standard deviation from a random
    sample of the dataset images (no masks, no augmentation).

    Uses a single-pass accumulation of Σx and Σx² over all pixels, so
    memory cost is O(1) regardless of dataset size.

    Args:
        data_root   : path to BUSI root (handles both layout variants).
        img_size    : images are resized to (img_size × img_size) before
                      accumulation so that the statistics match the input
                      resolution actually used during training.
        classes     : which class sub-directories to scan.
        max_samples : maximum number of images to use (random sample if
                      the dataset is larger).
        seed        : random seed for the subsample selection.

    Returns:
        (mean, std)  where each is a 3-tuple (R, G, B) of float values
        in [0, 1].  Falls back to ImageNet defaults if no images are found.
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

    # Single-pass accumulation: E[x] and E[x²] → mean, std
    sum_   = np.zeros(3, dtype=np.float64)
    sum2_  = np.zeros(3, dtype=np.float64)
    n_pix  = 0

    for path in sample:
        img = np.array(
            Image.open(path).convert("RGB").resize(
                (img_size, img_size), Image.BILINEAR),
            dtype=np.float64
        ) / 255.0                         # (H, W, 3) in [0, 1]
        pixels  = img.reshape(-1, 3)      # (H*W, 3)
        sum_   += pixels.sum(axis=0)
        sum2_  += (pixels ** 2).sum(axis=0)
        n_pix  += pixels.shape[0]

    mean = sum_ / n_pix
    var  = sum2_ / n_pix - mean ** 2
    std  = np.sqrt(np.maximum(var, 1e-8))  # guard against numerical noise

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
        
        # --- PREPROCESS CHUYÊN DỤNG CHO SIÊU ÂM ---
        # Tăng cường độ tương phản cục bộ (giúp làm rõ viền khối u)
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        
        # --- AUGMENTATION ---
        A.HorizontalFlip(p=0.5),
        # Ảnh vú siêu âm thường không lật ngược từ trên xuống dưới trong thực tế y khoa, 
        # nhưng nếu tập dữ liệu nhỏ bạn có thể giữ VerticalFlip(p=0.2)
        A.VerticalFlip(p=0.2), 
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15,
                           rotate_limit=30, p=0.5, border_mode=0), # border_mode=0 để thêm viền đen thay vì nội suy
        
        A.ElasticTransform(alpha=120, sigma=120 * 0.05,
                           alpha_affine=120 * 0.03, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.4),
        
        # Đã sửa: per_channel=False để không bị nhiễu màu cầu vồng. 
        # Giảm var_limit xuống mức hợp lý để không phá huỷ ảnh.
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
        
        # Validation cũng bắt buộc phải đi qua bước Preprocess giống Train
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        
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
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# ================================================================
# BUSI DATASET
# ================================================================
class BUSIDataset(Dataset):
    """
    BUSI (Breast Ultrasound Images) Dataset.

    Expected directory structure (either variant is handled):
      Variant A (flat per-class folders):
        data_root/benign/benign (1).png
        data_root/benign/benign (1)_mask.png
        data_root/benign/benign (1)_mask_1.png   ← extra lesion, handled!

      Variant B (Dataset_BUSI_with_GT layout):
        data_root/Dataset_BUSI_with_GT/benign/benign (1).png
        data_root/Dataset_BUSI_with_GT/benign/benign (1)_mask.png

    Multi-mask support
    ------------------
    Some BUSI images contain more than one lesion, each annotated as a
    separate mask file:
        benign (100)_mask.png     ← primary mask
        benign (100)_mask_1.png   ← additional lesion
        benign (100)_mask_2.png   ← yet another lesion

    All sibling masks for an image are merged with a pixel-wise maximum
    (logical OR), producing a single binary mask that covers every
    annotated lesion region.  This prevents silently ignoring lesions and
    matches clinical practice where all lesions should be segmented.
    """

    def __init__(
        self,
        data_root:    str,
        split:        str   = "train",             # "train" | "val" | "test"
        transforms:   Optional[A.Compose] = None,
        classes:      Tuple[str, ...] = ("benign", "malignant"),
        train_ratio:  float = 0.70,
        val_ratio:    float = 0.15,
        seed:         int   = 42,
    ):
        self.transforms = transforms
        # Each entry: (img_path: str, mask_paths: List[str])
        self.samples: List[Tuple[str, List[str]]] = []

        root = Path(data_root)
        # Handle Variant B layout
        candidate = root / "Dataset_BUSI_with_GT"
        if candidate.exists():
            root = candidate

        all_pairs: List[Tuple[str, List[str]]] = []
        total_multi = 0   # count images with more than one mask (diagnostic)

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
                    if len(mask_paths) > 1:
                        total_multi += 1
                    all_pairs.append((str(img_path),
                                      [str(m) for m in mask_paths]))
                else:
                    print(f"  [WARN] mask not found for {img_path.name}")

        if len(all_pairs) == 0:
            raise FileNotFoundError(
                f"No image-mask pairs found in {root}. "
                "Check that BUSI data is extracted correctly."
            )

        # Deterministic split
        rng = random.Random(seed)
        rng.shuffle(all_pairs)
        n = len(all_pairs)
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

        # ── Multi-mask merge (pixel-wise maximum = logical OR) ──────────────
        # Primary mask is always mask_paths[0].  Additional lesion masks
        # (mask_paths[1], mask_paths[2], …) are merged in so that all
        # annotated regions contribute equally to the training signal.
        mask = np.zeros(
            np.array(Image.open(mask_paths[0]).convert("L"),
                     dtype=np.uint8).shape,
            dtype=np.uint8,
        )
        for mp in mask_paths:
            m    = np.array(Image.open(mp).convert("L"), dtype=np.uint8)
            mask = np.maximum(mask, m)

        # Binarise (some BUSI masks store 0 / 255 instead of 0 / 1)
        mask = (mask > 127).astype(np.uint8)

        if self.transforms:
            aug  = self.transforms(image=img, mask=mask)
            img  = aug["image"]                          # (3, H, W) float tensor
            mask = aug["mask"].unsqueeze(0).float()      # (1, H, W)
        else:
            img  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask[None]).float()

        return img, mask


# ================================================================
# PRIVATE HELPER — find all mask files for one image
# ================================================================
def _find_all_masks(img_path: Path) -> List[Path]:
    """
    Return all mask files associated with *img_path*, sorted consistently.

    Searches for:
        <stem>_mask<suffix>          → primary mask  (always first if present)
        <stem>_mask_1<suffix>        → 1st extra lesion
        <stem>_mask_2<suffix>        → 2nd extra lesion
        … up to _mask_9 (covers all known BUSI variants)

    Example
    -------
        img_path = /data/benign/benign (100).png
        returns  [benign (100)_mask.png,
                  benign (100)_mask_1.png]   ← if the second file exists
    """
    stem   = img_path.stem    # e.g. "benign (100)"
    suffix = img_path.suffix  # e.g. ".png"
    found: List[Path] = []

    # Primary mask
    primary = img_path.with_name(stem + "_mask" + suffix)
    if primary.exists():
        found.append(primary)

    # Additional lesion masks: _mask_1, _mask_2, …
    for k in range(1, 10):
        extra = img_path.with_name(stem + f"_mask_{k}" + suffix)
        if extra.exists():
            found.append(extra)
        # Mask files are numbered consecutively, so stop on first miss
        else:
            break

    return found


# ================================================================
# DATALOADER BUILDER
# ================================================================
def build_dataloaders(cfg: Config):
    """
    Return (train_loader, val_loader, test_loader).

    If cfg.img_mean / cfg.img_std are None (the default), per-channel
    mean and standard deviation are computed automatically from a random
    sample of the training images and stored back into cfg so that
    the same statistics can be re-used for inference without recomputing.
    """
    # ── Compute or use supplied normalisation statistics ─────────────────────
    if cfg.img_mean is None or cfg.img_std is None:
        print("  Computing dataset normalisation statistics …")
        mean, std = compute_dataset_stats(
            data_root   = cfg.data_root,
            img_size    = cfg.img_size,
            classes     = cfg.use_classes,
            max_samples = 500,
            seed        = cfg.seed,
        )
        cfg.img_mean = mean
        cfg.img_std  = std
    else:
        mean, std = cfg.img_mean, cfg.img_std

    # ── Build transforms with the chosen statistics ───────────────────────────
    train_tf = get_train_transforms(cfg.img_size, mean, std)
    val_tf   = get_val_transforms(cfg.img_size, mean, std)

    train_ds = BUSIDataset(cfg.data_root, "train", train_tf,
                           cfg.use_classes, cfg.train_split,
                           cfg.val_split, cfg.seed)
    val_ds   = BUSIDataset(cfg.data_root, "val",   val_tf,
                           cfg.use_classes, cfg.train_split,
                           cfg.val_split, cfg.seed)
    test_ds  = BUSIDataset(cfg.data_root, "test",  val_tf,
                           cfg.use_classes, cfg.train_split,
                           cfg.val_split, cfg.seed)

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
