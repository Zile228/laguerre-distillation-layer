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

Metrics logged per epoch  : Dice, IoU, HD95, train loss
Operational metrics (once): GFLOPs, #Params, Latency (ms), GPU memory

Fix log:
  • torch.nn.functional (F) moved to top-level import — it was mistakenly
    re-imported inside the per-epoch LDL mass-update block, which shadowed
    the module-level name and is a latent bug.

  • LDL warmup order (Issue 5) — the original code called warmup_anchors()
    before freeze_psi_T(), triggering a RuntimeError because k-means++ was
    seeded with random (untrained) ψ_T embeddings that would then drift once
    the real warm-up began.

    Correct order (implemented below in run_method / _ldl_warmup_psi_T):
      1. Train ψ_T for cfg.psi_T_warmup_steps gradient steps with a
         variance-maximising objective so it learns a stable, non-collapsed
         embedding before anchors are seeded.
      2. Call ldl.freeze_psi_T()  → sets _psi_T_frozen = True.
      3. Collect psi_T-projected features from the training loader.
      4. Call ldl.warmup_anchors(feat_batches) → k-means++ init.
      5. Begin regular joint training.
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
import torch.nn.functional as F          # top-level import (was re-imported
                                          # inside loop — moved here)
from typing import Optional
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

def dice_score(
    pred_logit: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    pred  = (torch.sigmoid(pred_logit) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (denom + eps)).mean().item()


def iou_score(
    pred_logit: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    pred  = (torch.sigmoid(pred_logit) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


class HD95Meter:
    """Accumulates HD95 across batches using MONAI."""

    def __init__(self, device):
        if not MONAI_OK:
            self._ok = False
            return
        self._ok    = True
        self.metric = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean"
        )
        self.binarize = AsDiscrete(threshold=0.5)
        self.device   = device

    def update(self, pred_logit: torch.Tensor, target: torch.Tensor):
        if not self._ok:
            return
        pred    = (torch.sigmoid(pred_logit) > 0.5).long()
        tgt     = target.long()
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

def benchmark_model(
    model: nn.Module,
    cfg: Config,
    label: str = "model",
) -> dict:
    """Returns GFLOPs, params, latency_ms, gpu_mem_mb."""
    model.eval()
    dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=cfg.device)

    # ── Parameter count ───────────────────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # ── GFLOPs ────────────────────────────────────────────────────────────────
    gflops = float("nan")
    if THOP_OK:
        try:
            macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
            gflops  = macs * 2 / 1e9
        except Exception as e:
            print(f"  [WARN] thop failed for {label}: {e}")

    # ── Latency ───────────────────────────────────────────────────────────────
    warmup    = cfg.benchmark_warmup
    repeats   = cfg.benchmark_repeats
    latencies = []
    with torch.no_grad():
        for i in range(warmup + repeats):
            if cfg.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _  = model(dummy)
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
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6

    result = {
        "label":          label,
        "params_M":       round(n_params, 2),
        "gflops":         round(gflops, 3) if not np.isnan(gflops) else "N/A",
        "latency_ms":     round(lat_mean, 2),
        "latency_std_ms": round(lat_std, 2),
        "gpu_mem_mb":     round(gpu_mem, 1) if not np.isnan(gpu_mem) else "N/A",
    }
    print(f"  [{label}] params={result['params_M']}M | "
          f"GFLOPs={result['gflops']} | "
          f"latency={result['latency_ms']}±{result['latency_std_ms']}ms | "
          f"GPU mem={result['gpu_mem_mb']}MB")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# LDL-specific helper: ψ_T warm-up  (Issue 5 fix)
# ═══════════════════════════════════════════════════════════════════════════════

def _ldl_warmup_psi_T(
    kd_loss_fn,
    teacher:      nn.Module,
    train_loader,
    cfg:          Config,
) -> None:
    """
    Warm up the teacher adapter ψ_T for cfg.psi_T_warmup_steps gradient
    steps before freezing it and seeding the anchors via k-means++.

    Why this matters (Issue 5)
    --------------------------
    ψ_T is a 1×1 conv that projects teacher features into the D-dimensional
    embedding space shared with the anchors.  If k-means++ is run on random
    (untrained) ψ_T embeddings, the anchor positions are meaningless and then
    become mis-aligned with the embedding space once ψ_T is updated further.
    The correct sequence is:

        ψ_T warms up  →  freeze ψ_T  →  k-means++ on frozen embeddings

    Warm-up objective
    -----------------
    We maximise the per-channel variance of the L2-normalised embeddings.
    This encourages ψ_T to spread its output across the unit hypersphere
    (avoiding representational collapse to a single point) and is equivalent
    to minimising the negative mean variance, a standard anti-collapse loss.

    If cfg.psi_T_warmup_steps == 0 the warm-up is skipped (useful for
    unit tests and tiny datasets where random initialisation is acceptable).
    """
    warmup_steps = cfg.psi_T_warmup_steps
    if warmup_steps <= 0:
        print("  [LDL] psi_T_warmup_steps=0 — skipping ψ_T warm-up.")
        kd_loss_fn.freeze_psi_T()
        return

    print(f"  [LDL] Warming up ψ_T for {warmup_steps} gradient steps …")
    psi_T_opt = optim.Adam(kd_loss_fn.psi_T.parameters(), lr=cfg.lr * 0.1)

    teacher.eval()
    step = 0
    done = False

    while not done:
        for imgs, _ in train_loader:
            if step >= warmup_steps:
                done = True
                break

            imgs = imgs.to(cfg.device, non_blocking=True)

            with torch.no_grad():
                _, ft = get_encoder_feat(teacher, imgs, cfg.distill_feat_idx)

            psi_T_opt.zero_grad()

            ft_emb  = kd_loss_fn.psi_T(ft.detach())         # (B, D, H, W)
            B, D, H, W = ft_emb.shape
            ft_flat = ft_emb.permute(0, 2, 3, 1).reshape(-1, D)   # (B·H·W, D)
            ft_norm = F.normalize(ft_flat, dim=1)

            # Variance-maximising loss: −mean(per-channel variance)
            # Encourages spread across the hypersphere; prevents collapse.
            loss_warmup = -ft_norm.var(dim=0).mean()
            loss_warmup.backward()
            psi_T_opt.step()
            step += 1

    # ── Phase B: freeze ψ_T ──────────────────────────────────────────────────
    kd_loss_fn.freeze_psi_T()

    # ── Phase C: k-means++ anchor initialisation on frozen ψ_T ──────────────
    n_collect = min(10, len(train_loader))
    print(f"  [LDL] Collecting features from {n_collect} batches "
          f"for k-means++ anchor init …")
    teacher.eval()
    warmup_feats = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(train_loader):
            if i >= n_collect:
                break
            imgs    = imgs.to(cfg.device, non_blocking=True)
            _, ft   = get_encoder_feat(teacher, imgs, cfg.distill_feat_idx)
            ft_proj = kd_loss_fn.psi_T(ft).detach().cpu()
            warmup_feats.append(ft_proj)

    kd_loss_fn.warmup_anchors(warmup_feats)


# ═══════════════════════════════════════════════════════════════════════════════
# LDL per-epoch mass update helper
# ═══════════════════════════════════════════════════════════════════════════════

def _ldl_update_masses(
    kd_loss_fn,
    teacher: nn.Module,
    val_loader,
    cfg: Config,
) -> None:
    """
    Recompute anchor masses m_i ← |C_i(w) ∩ Ω| / |Ω| using the validation
    loader (Theorem 3.5).  Called once per epoch before training starts.
    Uses the validation set rather than the train set to avoid stale
    augmented views inflating or deflating cell volumes.
    """
    teacher.eval()
    C_accum: Optional[torch.Tensor] = None
    cnt = 0

    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs    = imgs.to(cfg.device, non_blocking=True)
            _, ft   = get_encoder_feat(teacher, imgs, cfg.distill_feat_idx)
            ft_proj = kd_loss_fn.psi_T(ft)

            B, _, H, W = ft.shape
            N = H * W

            grid_h = torch.linspace(0, 1, H, device=cfg.device)
            grid_w = torch.linspace(0, 1, W, device=cfg.device)
            gy, gx = torch.meshgrid(grid_h, grid_w, indexing="ij")
            P      = (torch.stack([gy, gx], dim=-1)
                      .reshape(N, 2).unsqueeze(0).expand(B, -1, -1))

            ft_flat = ft_proj.permute(0, 2, 3, 1).reshape(B, N, kd_loss_fn.D)
            F_hat   = torch.cat([ft_flat, P], dim=-1)          # (B, N, D+2)
            C_batch = kd_loss_fn._cost_matrix(F_hat).mean(0)   # (N, M)

            C_accum = C_batch if C_accum is None else C_accum + C_batch
            cnt += 1

    if C_accum is not None and cnt > 0:
        kd_loss_fn.update_masses(C_accum / cnt)


# ═══════════════════════════════════════════════════════════════════════════════
# One epoch of training
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    teacher,
    student,
    kd_loss_fn,
    seg_loss_fn,
    optimizer,
    scaler,
    loader,
    cfg,
    method,
):
    student.train()
    if teacher is not None:
        teacher.eval()

    total_loss = total_seg = total_kd = 0.0
    n_batches  = 0

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
                l_kd  = torch.tensor(0.0, device=cfg.device)
                loss  = l_seg
            elif method == "vanilla":
                l_kd  = kd_loss_fn(s_logit, t_logit)
                loss  = l_seg + cfg.lambda_kd * l_kd
            elif method in ("fitnets", "at"):
                l_kd  = kd_loss_fn(s_feat, t_feat)
                loss  = l_seg + cfg.lambda_kd * l_kd
            elif method == "ldl":
                l_kd  = kd_loss_fn(t_feat, s_feat)
                loss  = l_seg + cfg.lambda_kd * l_kd
            else:
                l_kd  = torch.tensor(0.0, device=cfg.device)
                loss  = l_seg

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
        "loss": total_loss / n_batches,
        "seg":  total_seg  / n_batches,
        "kd":   total_kd   / n_batches,
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
            loss      = seg_loss_fn(logit, masks)

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

    # ── Determine feature channel widths ──────────────────────────────────────
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

    # ── LDL initialisation: correct three-phase warmup (Issue 5 fix) ─────────
    #
    #   OLD (broken):
    #       collect features with random ψ_T  →  warmup_anchors()
    #       → RuntimeError because _psi_T_frozen is still False
    #
    #   NEW (correct):
    #       Phase A: train ψ_T for psi_T_warmup_steps steps (variance-max loss)
    #       Phase B: freeze_psi_T()              sets _psi_T_frozen = True
    #       Phase C: collect features, warmup_anchors()  (k-means++ on stable ψ_T)
    #
    if method == "ldl" and kd_loss_fn is not None:
        _ldl_warmup_psi_T(kd_loss_fn, teacher, train_loader, cfg)

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
    csv_path   = os.path.join(cfg.log_dir, f"{method}_history.csv")
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "train_loss", "train_seg", "train_kd",
        "val_loss", "val_dice", "val_iou", "val_hd95",
    ])
    csv_writer.writeheader()

    best_dice  = 0.0
    best_ckpt  = os.path.join(cfg.ckpt_dir, f"best_{method}.pth")
    no_improve = 0
    history    = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, cfg.epochs + 1):

        # LDL: update anchor masses once per epoch after the first
        if method == "ldl" and kd_loss_fn is not None and epoch > 1:
            _ldl_update_masses(kd_loss_fn, teacher, val_loader, cfg)

        train_metrics = train_one_epoch(
            teacher, student, kd_loss_fn, seg_loss_fn,
            optimizer, scaler, train_loader, cfg, method,
        )
        val_metrics = evaluate(student, val_loader, cfg)
        scheduler.step(val_metrics["dice"])

        hd95_str = (f"{val_metrics['hd95']:.3f}"
                    if not np.isnan(val_metrics["hd95"]) else "N/A")
        row = {
            "epoch":      epoch,
            "train_loss": round(train_metrics["loss"], 5),
            "train_seg":  round(train_metrics["seg"],  5),
            "train_kd":   round(train_metrics["kd"],   5),
            "val_loss":   round(val_metrics["loss"],   5),
            "val_dice":   round(val_metrics["dice"],   5),
            "val_iou":    round(val_metrics["iou"],    5),
            "val_hd95":   hd95_str,
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

        # ── Checkpoint ────────────────────────────────────────────────────────
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
          f"iou={test_metrics['iou']:.4f} | "
          f"hd95={test_metrics['hd95']:.2f}")

    # ── Operational benchmarks ────────────────────────────────────────────────
    print("\n  --- Operational Benchmarks ---")
    ops_student = benchmark_model(student, cfg, label=f"student_{method}")
    if method == "none":   # benchmark teacher only once
        ops_teacher = benchmark_model(teacher, cfg, label="teacher")
    else:
        ops_teacher = None

    return {
        "method":        method,
        "best_val_dice": round(best_dice, 4),
        "test_dice":     round(test_metrics["dice"], 4),
        "test_iou":      round(test_metrics["iou"],  4),
        "test_hd95":     (round(test_metrics["hd95"], 3)
                          if not np.isnan(test_metrics["hd95"]) else "N/A"),
        **{f"student_{k}": v for k, v in ops_student.items()
           if k != "label"},
        "teacher_ops":   ops_teacher,
        "history_csv":   csv_path,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison table printer
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(results: list):
    print("\n" + "=" * 90)
    print("  COMPARISON TABLE")
    print("=" * 90)
    header = (f"{'Method':<12} {'Dice':>6} {'IoU':>6} {'HD95':>7} "
              f"{'Params(M)':>10} {'GFLOPs':>8} {'Lat(ms)':>9} {'GPU(MB)':>8}")
    print(header)
    print("-" * 90)
    for r in results:
        print(f"{r['method']:<12} "
              f"{r['test_dice']:>6.4f} "
              f"{r['test_iou']:>6.4f} "
              f"{str(r['test_hd95']):>7} "
              f"{r['student_params_M']:>10} "
              f"{str(r['student_gflops']):>8} "
              f"{r['student_latency_ms']:>9.1f} "
              f"{str(r['student_gpu_mem_mb']):>8}")
    print("=" * 90)

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
    p.add_argument("--data_root",  default="./BUSI",
                   help="Path to BUSI dataset root")
    p.add_argument("--epochs",     type=int,   default=None,
                   help="Override Config.epochs")
    p.add_argument("--batch_size", type=int,   default=None,
                   help="Override Config.batch_size")
    p.add_argument("--lambda_kd",  type=float, default=None,
                   help="Override Config.lambda_kd")
    p.add_argument("--benchmark_only", action="store_true",
                   help="Only benchmark saved checkpoints (skip training)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    cfg           = Config()
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
        results = []
        for m in ALL_METHODS:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"best_{m}.pth")
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] No checkpoint for {m}")
                continue
            student = build_student(cfg)
            ckpt    = torch.load(ckpt_path, map_location=cfg.device)
            student.load_state_dict(ckpt["student"])
            ops = benchmark_model(student, cfg, label=f"student_{m}")
            _, _, test_loader = build_dataloaders(cfg)
            tm = evaluate(student, test_loader, cfg)
            results.append({
                "method":    m,
                "test_dice": round(tm["dice"], 4),
                "test_iou":  round(tm["iou"],  4),
                "test_hd95": (round(tm["hd95"], 3)
                              if not np.isnan(tm["hd95"]) else "N/A"),
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
