"""
train.py — Training, Evaluation, and Benchmarking Runner

Usage:
    python train.py --method ldl      # LDL distillation
    python train.py --method vanilla  # Vanilla KD
    python train.py --method fitnets  # FitNets
    python train.py --method at       # Attention Transfer
    python train.py --method none     # Student standalone
    python train.py --method all      # Run all methods sequentially
    python train.py --benchmark_only  # Skip training; just benchmark checkpoints

Metrics logged per epoch  : Dice, IoU, HD95, BCE loss
Operational metrics (once): GFLOPs, #Params, Latency (ms), GPU memory
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# ── Monai for HD95 ────────────────────────────────────────────────────────────
try:
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    from monai.transforms import AsDiscrete
    MONAI_OK = True
except ImportError:
    MONAI_OK = False
    print("[WARN] monai not found — HD95 will be skipped. pip install monai")

# ── THOP for FLOPs ───────────────────────────────────────────────────────────
try:
    from thop import profile as thop_profile
    THOP_OK = True
except ImportError:
    THOP_OK = False
    print("[WARN] thop not found — GFLOPs will be skipped. pip install thop")

from config_and_dataset_v2 import Config, build_dataloaders
from models_and_kd_v2 import (
    build_teacher, build_student, build_kd_loss,
    get_encoder_feat, SegLoss,
)

# ── Optional LDL import ───────────────────────────────────────────────────────
try:
    from ldl_layer_v2 import LaguerreDistillationLayer
except ImportError:
    LaguerreDistillationLayer = None


# ═══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def dice_score(pred_logit: torch.Tensor, target: torch.Tensor,
               threshold: float = 0.5, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(pred_logit) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (denom + eps)).mean().item()


def iou_score(pred_logit: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(pred_logit) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


class HD95Meter:
    """Accumulates HD95 across batches using MONAI."""
    def __init__(self, device):
        if not MONAI_OK:
            self._ok = False
            return
        self._ok = True
        self.metric = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean"
        )
        self.binarize = AsDiscrete(threshold=0.5)
        self.device = device

    def update(self, pred_logit: torch.Tensor, target: torch.Tensor):
        if not self._ok:
            return
        pred = (torch.sigmoid(pred_logit) > 0.5).long()
        tgt  = target.long()
        # MONAI expects (B, C, H, W) with one-hot classes
        pred_oh = torch.cat([1 - pred, pred], dim=1)
        tgt_oh  = torch.cat([1 - tgt,  tgt],  dim=1)
        self.metric(pred_oh.cpu(), tgt_oh.cpu())

    def compute(self) -> float:
        if not self._ok:
            return float("nan")
        val = self.metric.aggregate().item()
        self.metric.reset()
        return val


# ═══════════════════════════════════════════════════════════════════════════════
# Operational benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_model(model: nn.Module, cfg: Config,
                    label: str = "model") -> dict:
    """Returns GFLOPs, params, latency_ms, gpu_mem_mb."""
    model.eval()
    dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size,
                        device=cfg.device)

    # ── Parameter count ───────────────────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters()) / 1e6  # millions

    # ── GFLOPs ────────────────────────────────────────────────────────────────
    gflops = float("nan")
    if THOP_OK:
        try:
            macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
            gflops = macs * 2 / 1e9
        except Exception as e:
            print(f"  [WARN] thop failed for {label}: {e}")

    # ── Latency ───────────────────────────────────────────────────────────────
    warmup   = cfg.benchmark_warmup
    repeats  = cfg.benchmark_repeats
    latencies = []
    with torch.no_grad():
        for i in range(warmup + repeats):
            if cfg.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if cfg.device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= warmup:
                latencies.append((t1 - t0) * 1000)

    lat_mean = float(np.mean(latencies))
    lat_std  = float(np.std(latencies))

    # ── GPU memory ────────────────────────────────────────────────────────────
    gpu_mem = float("nan")
    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy)
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6  # MB

    result = {
        "label":         label,
        "params_M":      round(n_params, 2),
        "gflops":        round(gflops, 3) if not np.isnan(gflops) else "N/A",
        "latency_ms":    round(lat_mean, 2),
        "latency_std_ms": round(lat_std, 2),
        "gpu_mem_mb":    round(gpu_mem, 1) if not np.isnan(gpu_mem) else "N/A",
    }
    print(f"  [{label}] params={result['params_M']}M | "
          f"GFLOPs={result['gflops']} | "
          f"latency={result['latency_ms']}±{result['latency_std_ms']}ms | "
          f"GPU mem={result['gpu_mem_mb']}MB")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# One epoch of training
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(teacher, student, kd_loss_fn, seg_loss_fn,
                    optimizer, scaler, loader, cfg, method):
    student.train()
    if teacher is not None:
        teacher.eval()

    total_loss = total_seg = total_kd = 0.0
    n_batches = 0

    for imgs, masks in loader:
        imgs  = imgs.to(cfg.device,  non_blocking=True)
        masks = masks.to(cfg.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.amp):
            if teacher is not None:
                with torch.no_grad():
                    t_logit, t_feat = get_encoder_feat(
                        teacher, imgs, cfg.distill_feat_idx)
                s_logit, s_feat = get_encoder_feat(
                    student, imgs, cfg.distill_feat_idx)
            else:
                s_logit, s_feat = get_encoder_feat(
                    student, imgs, cfg.distill_feat_idx)
                t_logit = t_feat = None

            l_seg = seg_loss_fn(s_logit, masks)

            if kd_loss_fn is None or teacher is None:
                l_kd   = torch.tensor(0.0, device=cfg.device)
                loss   = l_seg
            elif method == "vanilla":
                l_kd = kd_loss_fn(s_logit, t_logit)
                loss = l_seg + cfg.lambda_kd * l_kd
            elif method in ("fitnets", "at"):
                l_kd = kd_loss_fn(s_feat, t_feat)
                loss = l_seg + cfg.lambda_kd * l_kd
            elif method == "ldl":
                l_kd = kd_loss_fn(t_feat, s_feat)
                loss = l_seg + cfg.lambda_kd * l_kd
            else:
                l_kd = torch.tensor(0.0, device=cfg.device)
                loss = l_seg

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_seg  += l_seg.item()
        total_kd   += l_kd.item() if isinstance(l_kd, torch.Tensor) else l_kd
        n_batches  += 1

    return {
        "loss":    total_loss / n_batches,
        "seg":     total_seg  / n_batches,
        "kd":      total_kd   / n_batches,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(student, loader, cfg):
    student.eval()
    seg_loss_fn = SegLoss().to(cfg.device)
    hd95_meter  = HD95Meter(cfg.device)

    total_dice = total_iou = total_loss = 0.0
    n = 0

    for imgs, masks in loader:
        imgs  = imgs.to(cfg.device,  non_blocking=True)
        masks = masks.to(cfg.device, non_blocking=True)

        with autocast(enabled=cfg.amp):
            logit, _ = get_encoder_feat(student, imgs, -2)
            loss = seg_loss_fn(logit, masks)

        total_dice += dice_score(logit, masks)
        total_iou  += iou_score(logit, masks)
        total_loss += loss.item()
        hd95_meter.update(logit, masks)
        n += 1

    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou":  total_iou  / n,
        "hd95": hd95_meter.compute(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Full training run for one method
# ═══════════════════════════════════════════════════════════════════════════════

def run_method(method: str, cfg: Config):
    print(f"\n{'='*60}")
    print(f"  METHOD: {method.upper()}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # ── Build models ──────────────────────────────────────────────────────────
    teacher = build_teacher(cfg)
    student = build_student(cfg)

    # ── Determine feature channel widths (needed for some KD losses) ─────────
    with torch.no_grad():
        dummy = torch.randn(2, 3, cfg.img_size, cfg.img_size,
                            device=cfg.device)
        _, t_feat_sample = get_encoder_feat(teacher, dummy, cfg.distill_feat_idx)
        _, s_feat_sample = get_encoder_feat(student, dummy, cfg.distill_feat_idx)
    teacher_ch = t_feat_sample.shape[1]
    student_ch = s_feat_sample.shape[1]
    print(f"  Teacher distill channels: {teacher_ch}")
    print(f"  Student distill channels: {student_ch}")

    # ── KD loss module ────────────────────────────────────────────────────────
    kd_loss_fn  = build_kd_loss(method, cfg, teacher_ch, student_ch)
    seg_loss_fn = SegLoss().to(cfg.device)

    # ── LDL warmup: k-means++ anchor init ────────────────────────────────────
    if method == "ldl" and kd_loss_fn is not None:
        print("  [LDL] Running k-means++ warmup on first 5 train batches …")
        teacher.eval()
        warmup_feats = []
        with torch.no_grad():
            for i, (imgs, _) in enumerate(train_loader):
                if i >= 5:
                    break
                imgs = imgs.to(cfg.device)
                _, ft = get_encoder_feat(teacher, imgs, cfg.distill_feat_idx)
                ft_proj = kd_loss_fn.psi_T(ft).detach().cpu()
                warmup_feats.append(ft_proj)
        kd_loss_fn.warmup_anchors(warmup_feats)

    # ── Optimiser: student params + (optionally) KD params ───────────────────
    params = list(student.parameters())
    if kd_loss_fn is not None:
        params += list(kd_loss_fn.parameters())
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=cfg.lr_patience, factor=0.5
    )
    scaler = GradScaler(enabled=cfg.amp)

    # ── CSV logger ────────────────────────────────────────────────────────────
    csv_path = os.path.join(cfg.log_dir, f"{method}_history.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "train_loss", "train_seg", "train_kd",
        "val_loss", "val_dice", "val_iou", "val_hd95"
    ])
    csv_writer.writeheader()

    best_dice     = 0.0
    best_ckpt     = os.path.join(cfg.ckpt_dir, f"best_{method}.pth")
    no_improve    = 0
    history       = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, cfg.epochs + 1):

        # LDL mass update once per epoch
        if method == "ldl" and kd_loss_fn is not None and epoch > 1:
            teacher.eval()
            C_accum = None
            cnt = 0
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs = imgs.to(cfg.device)
                    _, ft = get_encoder_feat(teacher, imgs, cfg.distill_feat_idx)
                    ft_proj = kd_loss_fn.psi_T(ft)
                    B, _, H, W = ft.shape
                    N = H * W
                    import torch.nn.functional as F
                    grid_h = torch.linspace(0, 1, H, device=cfg.device)
                    grid_w = torch.linspace(0, 1, W, device=cfg.device)
                    gy, gx = torch.meshgrid(grid_h, grid_w, indexing="ij")
                    P = torch.stack([gy, gx], dim=-1).reshape(N, 2).unsqueeze(0).expand(B, -1, -1)
                    ft_flat = ft_proj.permute(0, 2, 3, 1).reshape(B, N, kd_loss_fn.D)
                    F_hat   = torch.cat([ft_flat, P], dim=-1)
                    C = kd_loss_fn._cost_matrix(F_hat).mean(0)  # (N, M)
                    C_accum = C if C_accum is None else C_accum + C
                    cnt += 1
            if C_accum is not None:
                kd_loss_fn.update_masses(C_accum / cnt)

        train_metrics = train_one_epoch(
            teacher, student, kd_loss_fn, seg_loss_fn,
            optimizer, scaler, train_loader, cfg, method
        )
        val_metrics = evaluate(student, val_loader, cfg)
        scheduler.step(val_metrics["dice"])

        row = {
            "epoch":      epoch,
            "train_loss": round(train_metrics["loss"], 5),
            "train_seg":  round(train_metrics["seg"],  5),
            "train_kd":   round(train_metrics["kd"],   5),
            "val_loss":   round(val_metrics["loss"],   5),
            "val_dice":   round(val_metrics["dice"],   5),
            "val_iou":    round(val_metrics["iou"],    5),
            "val_hd95":   round(val_metrics["hd95"],   3)
                          if not np.isnan(val_metrics["hd95"]) else "N/A",
        }
        csv_writer.writerow(row)
        csv_file.flush()
        history.append(row)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/{cfg.epochs} | "
                  f"loss={row['train_loss']:.4f} | "
                  f"val_dice={row['val_dice']:.4f} | "
                  f"val_iou={row['val_iou']:.4f} | "
                  f"val_hd95={row['val_hd95']}")

        # Checkpoint
        if val_metrics["dice"] > best_dice:
            best_dice  = val_metrics["dice"]
            no_improve = 0
            torch.save({
                "epoch":   epoch,
                "student": student.state_dict(),
                "kd":      kd_loss_fn.state_dict() if kd_loss_fn else None,
                "optim":   optimizer.state_dict(),
                "dice":    best_dice,
            }, best_ckpt)
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {cfg.early_stop} epochs)")
                break

    csv_file.close()
    print(f"  Best val Dice: {best_dice:.4f}  →  checkpoint: {best_ckpt}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=cfg.device)
    student.load_state_dict(ckpt["student"])
    test_metrics = evaluate(student, test_loader, cfg)
    print(f"  TEST → dice={test_metrics['dice']:.4f} | "
          f"iou={test_metrics['iou']:.4f} | hd95={test_metrics['hd95']:.2f}")

    # ── Operational benchmarks ────────────────────────────────────────────────
    print("\n  --- Operational Benchmarks ---")
    ops_student = benchmark_model(student, cfg, label=f"student_{method}")
    if method == "none":   # only benchmark teacher once
        ops_teacher = benchmark_model(teacher, cfg, label="teacher")
    else:
        ops_teacher = None

    return {
        "method":      method,
        "best_val_dice": round(best_dice, 4),
        "test_dice":   round(test_metrics["dice"], 4),
        "test_iou":    round(test_metrics["iou"],  4),
        "test_hd95":   round(test_metrics["hd95"], 3)
                       if not np.isnan(test_metrics["hd95"]) else "N/A",
        **{f"student_{k}": v for k, v in ops_student.items()
           if k != "label"},
        "teacher_ops": ops_teacher,
        "history_csv": csv_path,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison table printer
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(results: list):
    print("\n" + "="*90)
    print("  COMPARISON TABLE")
    print("="*90)
    header = (f"{'Method':<12} {'Dice':>6} {'IoU':>6} {'HD95':>7} "
              f"{'Params(M)':>10} {'GFLOPs':>8} {'Lat(ms)':>9} {'GPU(MB)':>8}")
    print(header)
    print("-"*90)
    for r in results:
        print(f"{r['method']:<12} "
              f"{r['test_dice']:>6.4f} "
              f"{r['test_iou']:>6.4f} "
              f"{str(r['test_hd95']):>7} "
              f"{r['student_params_M']:>10} "
              f"{str(r['student_gflops']):>8} "
              f"{r['student_latency_ms']:>9.1f} "
              f"{str(r['student_gpu_mem_mb']):>8}")
    print("="*90)

    # Save to JSON
    out_path = "./results/comparison_results.json"
    os.makedirs("./results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

ALL_METHODS = ["none", "vanilla", "fitnets", "at", "ldl"]


def parse_args():
    p = argparse.ArgumentParser(description="LDL KD Comparison Runner")
    p.add_argument("--method", default="ldl",
                   choices=ALL_METHODS + ["all"],
                   help="KD method to run (or 'all' for full comparison)")
    p.add_argument("--data_root", default="./BUSI",
                   help="Path to BUSI dataset root")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override Config.epochs")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override Config.batch_size")
    p.add_argument("--benchmark_only", action="store_true",
                   help="Only run operational benchmarks on saved checkpoints")
    p.add_argument("--lambda_kd", type=float, default=None,
                   help="Override Config.lambda_kd")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    cfg.data_root = args.data_root
    if args.epochs     is not None: cfg.epochs     = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.lambda_kd  is not None: cfg.lambda_kd  = args.lambda_kd
    cfg.seed = args.seed

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"\n  Device : {cfg.device}")
    print(f"  AMP    : {cfg.amp}")
    print(f"  Epochs : {cfg.epochs}")
    print(f"  Batch  : {cfg.batch_size}")

    if args.benchmark_only:
        # Quick benchmark loop on saved checkpoints
        cfg_b = Config()
        cfg_b.data_root = args.data_root
        results = []
        for m in ALL_METHODS:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"best_{m}.pth")
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] No checkpoint for {m}")
                continue
            student = build_student(cfg_b)
            ckpt    = torch.load(ckpt_path, map_location=cfg_b.device)
            student.load_state_dict(ckpt["student"])
            ops = benchmark_model(student, cfg_b, label=f"student_{m}")
            _, _, test_loader = build_dataloaders(cfg_b)
            tm = evaluate(student, test_loader, cfg_b)
            results.append({
                "method": m,
                "test_dice": round(tm["dice"], 4),
                "test_iou":  round(tm["iou"],  4),
                "test_hd95": round(tm["hd95"], 3)
                              if not np.isnan(tm["hd95"]) else "N/A",
                **{f"student_{k}": v for k, v in ops.items() if k != "label"},
                "teacher_ops": None,
            })
        if results:
            print_comparison_table(results)
        return

    methods = ALL_METHODS if args.method == "all" else [args.method]
    results = []
    for m in methods:
        r = run_method(m, cfg)
        results.append(r)

    if len(results) > 1:
        print_comparison_table(results)
    else:
        r = results[0]
        print(f"\n  Final test → dice={r['test_dice']} | "
              f"iou={r['test_iou']} | hd95={r['test_hd95']}")


if __name__ == "__main__":
    main()
