"""
train.py — Training, Evaluation, and Benchmarking Runner

Usage:
    python train.py --method ldl   --seed 42   # single run
    python train.py --method all   --seed 42   # all methods, one seed
    python train.py --method all   --seed 123  # all methods, second seed
    python train.py --benchmark_only            # aggregate all saved checkpoints

Multi-seed workflow (recommended):
    for seed in 42 123 456; do
        python train.py --method all --seed $seed
    done
    python train.py --benchmark_only   # produces mean ± std across seeds

Fix log (v2 → v3):
─────────────────────────────────────────────────────────────────────────────
NEW FIXES:
    - Added "teacher" to ALL_METHODS. Calling `--method all` will now dynamically 
      train the teacher architecture with the dataset and save the optimal weights, 
      eliminating the Random Teacher issue.
    - Repaired `dice_score` and `iou_score` accumulators to utilize a properly 
      scaled sum instead of the mean, preventing biased Evaluation Metric Aggregation.
    - Moved random seed instantiation (torch.manual_seed, etc.) immediately inside 
      run_method() to completely prevent RNG Seed Contamination. 
    - The OT Mass Calculation in `_ldl_update_masses` now dynamically isolates the 
      cost matrix logic per-image preventing validly scaled assignments from being
      smoothed over validation sets prior to the sub-gradient operation.

Fix 1  [Vanilla KD imbalance — train.py side]
    The T² scaling was removed inside VanillaKDLoss (models_and_kd_v2.py).
    No train.py change needed; the loss is now in a balanced range.

Fix 2  [Multi-seed support]
    --seed N  controls the random seed for weight initialisation and
    data-augmentation order.  Dataset splits are always seeded at 42 so
    the test set is identical across seeds.  Checkpoint and log files are
    named best_{method}_seed{seed}.pth / {method}_seed{seed}_history.csv.
    benchmark_only mode scans for all matching checkpoints, evaluates each,
    and reports mean ± std per method.

Fix 3  [Teacher test metrics]
    run_method() now evaluates the frozen teacher on the test set whenever
    method == 'none' and stores dice/iou/hd95 in the results dict alongside
    the operational benchmarks.  benchmark_only also evaluates the teacher
    once (it has no seed-to-seed variation).

Fix 4  [LDL hyperparameters — config side only]
    embed_dim=256 and num_anchors=64 set in Config; no train.py change.

Fix 5  [λ_LDL ramp-up]
    train_one_epoch() now accepts an epoch argument.  For the LDL method,
    the effective KD weight scales from 0 → λ_kd over the first
    cfg.ldl_lambda_ramp_epochs epochs after the ψ_T warm-up phase.
    This eliminates the abrupt loss spike previously observed at epoch 25.
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

try:
    from monai.metrics import HausdorffDistanceMetric
    MONAI_OK = True
except ImportError:
    MONAI_OK = False
    print("[WARN] monai not found — HD95 will be skipped.  pip install monai")

try:
    from thop import profile as thop_profile
    THOP_OK = True
except ImportError:
    THOP_OK = False
    print("[WARN] thop not found — GFLOPs will be skipped.  pip install thop")

from config_and_dataset_v2 import Config, build_dataloaders
from models_and_kd_v2 import (
    build_teacher, build_student, build_kd_loss,
    get_encoder_feat, SegLoss,
)

try:
    from ldl_layer_v2 import LaguerreDistillationLayer
except ImportError:
    LaguerreDistillationLayer = None


# ═══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def dice_score(pred_logit, target, threshold=0.5, eps=1e-6):
    pred  = (torch.sigmoid(pred_logit) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    # Corrected to sum over batch elements for unbiased dataset-level tracking
    return ((2 * inter + eps) / (denom + eps)).sum().item()


def iou_score(pred_logit, target, threshold=0.5, eps=1e-6):
    pred  = (torch.sigmoid(pred_logit) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    # Corrected to sum over batch elements for unbiased dataset-level tracking
    return ((inter + eps) / (union + eps)).sum().item()


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
        self.device = device

    def update(self, pred_logit, target):
        if not self._ok:
            return
        pred    = (torch.sigmoid(pred_logit) > 0.5).long()
        tgt     = target.long()
        pred_oh = torch.cat([1 - pred, pred], dim=1)
        tgt_oh  = torch.cat([1 - tgt,  tgt],  dim=1)
        self.metric(pred_oh.cpu(), tgt_oh.cpu())

    def compute(self):
        if not self._ok:
            return float("nan")
        val = self.metric.aggregate().item()
        self.metric.reset()
        return val


# ═══════════════════════════════════════════════════════════════════════════════
# Operational benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_model(model, cfg, label="model"):
    """Returns GFLOPs, params, latency_ms, gpu_mem_mb."""
    model.eval()
    dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=cfg.device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    gflops = float("nan")
    if THOP_OK:
        try:
            macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
            gflops  = macs * 2 / 1e9
        except Exception as e:
            print(f"  [WARN] thop failed for {label}: {e}")

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
        "latency_std_ms": round(lat_std,  2),
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

def _ldl_warmup_psi_T(kd_loss_fn, teacher, train_loader, cfg):
    """
    Three-phase initialisation (Issue 5 fix):
      Phase A: variance-maximising warm-up of ψ_T (~500 gradient steps)
      Phase B: freeze ψ_T
      Phase C: k-means++ anchor init on frozen ψ_T embeddings
    """
    warmup_steps = cfg.psi_T_warmup_steps
    if warmup_steps <= 0:
        print("  [LDL] psi_T_warmup_steps=0 — skipping ψ_T warm-up.")
        kd_loss_fn.freeze_psi_T()
        return

    print(f"  [LDL] Warming up ψ_T for {warmup_steps} gradient steps …")
    psi_T_opt = optim.Adam(kd_loss_fn.psi_T.parameters(), lr=cfg.lr * 0.1)

    teacher.eval()
    step, done = 0, False
    while not done:
        for imgs, _ in train_loader:
            if step >= warmup_steps:
                done = True
                break
            imgs = imgs.to(cfg.device, non_blocking=True)
            with torch.no_grad():
                _, ft = get_encoder_feat(teacher, imgs, cfg.distill_feat_idx)
            psi_T_opt.zero_grad()
            ft_emb  = kd_loss_fn.psi_T(ft.detach())
            B, D, H, W = ft_emb.shape
            ft_flat = ft_emb.permute(0, 2, 3, 1).reshape(-1, D)
            ft_norm = F.normalize(ft_flat, dim=1)
            loss_warmup = -ft_norm.var(dim=0).mean()
            loss_warmup.backward()
            psi_T_opt.step()
            step += 1

    kd_loss_fn.freeze_psi_T()

    n_collect = min(10, len(train_loader))
    print(f"  [LDL] Collecting features from {n_collect} batches "
          f"for k-means++ anchor init …")
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
# LDL per-epoch mass update
# ═══════════════════════════════════════════════════════════════════════════════

def _ldl_update_masses(kd_loss_fn, teacher, val_loader, cfg):
    """
    Recompute anchor masses m_i ← |C_i(w) ∩ Ω| / |Ω| (Theorem 3.5).
    Correctly aggregates per-image instance optimal assignments over the dataset.
    """
    teacher.eval()
    mass_accum = torch.zeros(kd_loss_fn.M, device=cfg.device)
    total_samples = 0
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
            F_hat   = torch.cat([ft_flat, P], dim=-1)
            
            C_batch = kd_loss_fn._cost_matrix(F_hat) # (B, N, M)
            V = kd_loss_fn.k * C_batch - kd_loss_fn.w.unsqueeze(0).unsqueeze(0)
            sigma = V.argmin(dim=2) # (B, N)
            
            for i in range(kd_loss_fn.M):
                # Calculate assigned percentage per image, then sum across batch
                mass_accum[i] += (sigma == i).float().mean(dim=1).sum()
            total_samples += B
            
    if total_samples > 0:
        new_masses = (mass_accum / total_samples).clamp(min=1e-4)
        kd_loss_fn.update_masses(new_masses)


# ═══════════════════════════════════════════════════════════════════════════════
# One epoch of training  (Fix 5: epoch param added for λ ramp-up)
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
    epoch: int = 1,           # Fix 5: used to compute ramp-up factor
    ldl_warmup_done: bool = True,  # Fix 5: ramp starts after warmup
):
    """
    Fix 5 — λ_LDL ramp-up:
        For LDL, the effective KD weight scales linearly from 0 → λ_kd over
        cfg.ldl_lambda_ramp_epochs epochs after the ψ_T warm-up completes.
        This prevents the abrupt loss spike seen in v2 when the full OT
        objective was switched on instantly at epoch 25.
    """
    student.train()
    if teacher is not None:
        teacher.eval()

    total_loss = total_seg = total_kd = 0.0
    n_samples  = 0

    for imgs, masks in loader:
        B = imgs.shape[0]
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
                l_kd         = torch.tensor(0.0, device=cfg.device)
                effective_lam = 0.0
                loss          = l_seg

            elif method == "vanilla":
                l_kd          = kd_loss_fn(s_logit, t_logit)
                effective_lam = cfg.lambda_kd
                loss          = l_seg + effective_lam * l_kd

            elif method in ("fitnets", "at"):
                l_kd          = kd_loss_fn(s_feat, t_feat)
                effective_lam = cfg.lambda_kd
                loss          = l_seg + effective_lam * l_kd

            elif method == "ldl":
                l_kd = kd_loss_fn(t_feat, s_feat)
                # Fix 5: linear ramp-up of λ_LDL after warm-up phase
                ramp_epochs = max(1, cfg.ldl_lambda_ramp_epochs)
                if ldl_warmup_done:
                    ramp_frac = min(1.0, epoch / ramp_epochs)
                else:
                    ramp_frac = 0.0
                effective_lam = cfg.lambda_kd * ramp_frac
                loss          = l_seg + effective_lam * l_kd

            else:
                l_kd          = torch.tensor(0.0, device=cfg.device)
                effective_lam = 0.0
                loss          = l_seg

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * B
        total_seg  += l_seg.item() * B
        l_kd_val = l_kd.item() if isinstance(l_kd, torch.Tensor) else l_kd
        total_kd   += l_kd_val * B
        n_samples  += B

    return {
        "loss": total_loss / n_samples,
        "seg":  total_seg  / n_samples,
        "kd":   total_kd   / n_samples,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    seg_loss_fn = SegLoss().to(cfg.device)
    hd95_meter  = HD95Meter(cfg.device)
    total_dice = total_iou = total_loss = 0.0
    n_samples = 0
    
    for imgs, masks in loader:
        B = imgs.shape[0]
        imgs  = imgs.to(cfg.device,  non_blocking=True)
        masks = masks.to(cfg.device, non_blocking=True)
        with autocast(enabled=cfg.amp):
            logit, _ = get_encoder_feat(model, imgs, -2)
            loss      = seg_loss_fn(logit, masks)
        total_dice += dice_score(logit, masks)
        total_iou  += iou_score(logit, masks)
        total_loss += loss.item() * B
        hd95_meter.update(logit, masks)
        n_samples += B
        
    return {
        "loss": total_loss / n_samples,
        "dice": total_dice / n_samples,
        "iou":  total_iou  / n_samples,
        "hd95": hd95_meter.compute(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Full training run for one method and one seed
# ═══════════════════════════════════════════════════════════════════════════════

def run_method(method: str, cfg: Config):
    """
    Train one (method, seed) pair.

    Checkpoints   : {ckpt_dir}/best_{method}_seed{seed}.pth
    Training log  : {log_dir}/{method}_seed{seed}_history.csv
    Per-seed JSON : {results_dir}/{method}_seed{seed}_metrics.json
    """
    seed = cfg.seed
    
    # Fix 3: RNG Reseeding to avoid global RNG contamination
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    print(f"\n{'='*65}")
    print(f"  METHOD: {method.upper()}   SEED: {seed}")
    print(f"{'='*65}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    if method == "teacher":
        teacher = None
        student = build_teacher(cfg, load_ckpt=False)
    else:
        teacher = build_teacher(cfg, load_ckpt=True)
        student = build_student(cfg)

    # Detect feature channel widths dynamically (architecture-agnostic)
    with torch.no_grad():
        dummy = torch.randn(2, 3, cfg.img_size, cfg.img_size, device=cfg.device)
        if teacher is not None:
            _, t_feat_sample = get_encoder_feat(teacher, dummy, cfg.distill_feat_idx)
            teacher_ch = t_feat_sample.shape[1]
        else:
            teacher_ch = 0
            
        _, s_feat_sample = get_encoder_feat(student, dummy, cfg.distill_feat_idx)
        student_ch = s_feat_sample.shape[1]
        
    print(f"  Teacher distill channels : {teacher_ch}")
    print(f"  Student distill channels : {student_ch}")

    kd_loss_fn  = build_kd_loss(method, cfg, teacher_ch, student_ch)
    seg_loss_fn = SegLoss().to(cfg.device)

    # LDL three-phase warm-up (Issue 5 fix)
    ldl_warmup_epoch = 0    # epoch from which the ramp starts (Fix 5)
    if method == "ldl" and kd_loss_fn is not None:
        _ldl_warmup_psi_T(kd_loss_fn, teacher, train_loader, cfg)
        ldl_warmup_epoch = 1   # ramp-up begins at epoch 1 of joint training

    params = list(student.parameters())
    if kd_loss_fn is not None:
        params += list(kd_loss_fn.parameters())
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=cfg.lr_patience, factor=0.5
    )
    scaler = GradScaler(enabled=cfg.amp)

    csv_path   = os.path.join(cfg.log_dir, f"{method}_seed{seed}_history.csv")
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "train_loss", "train_seg", "train_kd",
        "val_loss", "val_dice", "val_iou", "val_hd95",
    ])
    csv_writer.writeheader()

    best_dice  = 0.0
    best_ckpt  = os.path.join(cfg.ckpt_dir, f"best_{method}_seed{seed}.pth")
    no_improve = 0
    history    = []

    for epoch in range(1, cfg.epochs + 1):

        if method == "ldl" and kd_loss_fn is not None and epoch > 1:
            _ldl_update_masses(kd_loss_fn, teacher, val_loader, cfg)

        # Fix 5: pass epoch number for λ ramp-up computation
        # epoch_in_joint: how many epochs since joint LDL training started
        epoch_in_joint = epoch - ldl_warmup_epoch + 1

        train_metrics = train_one_epoch(
            teacher, student, kd_loss_fn, seg_loss_fn,
            optimizer, scaler, train_loader, cfg, method,
            epoch=epoch_in_joint,
            ldl_warmup_done=(ldl_warmup_epoch > 0),
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

    # ── Student test evaluation ───────────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=cfg.device)
    student.load_state_dict(ckpt["student"])
    test_metrics = evaluate(student, test_loader, cfg)
    print(f"  TEST → dice={test_metrics['dice']:.4f} | "
          f"iou={test_metrics['iou']:.4f} | "
          f"hd95={test_metrics['hd95']:.2f}")

    # Fix 3: Teacher test metrics (once per run — teacher has no seed variation)
    teacher_test = None
    if method == "none":
        print("  Evaluating frozen teacher on test set …")
        teacher_test = evaluate(teacher, test_loader, cfg)
        print(f"  TEACHER TEST → dice={teacher_test['dice']:.4f} | "
              f"iou={teacher_test['iou']:.4f} | "
              f"hd95={teacher_test['hd95']:.2f}")

    # ── Operational benchmarks ────────────────────────────────────────────────
    print("\n  --- Operational Benchmarks ---")
    ops_student = benchmark_model(student, cfg, label=f"student_{method}")
    ops_teacher = None
    if method == "none":
        ops_teacher = benchmark_model(teacher, cfg, label="teacher")

    result = {
        "method":        method,
        "seed":          seed,
        "best_val_dice": round(best_dice, 4),
        "test_dice":     round(test_metrics["dice"], 4),
        "test_iou":      round(test_metrics["iou"],  4),
        "test_hd95":     (round(test_metrics["hd95"], 3)
                          if not np.isnan(test_metrics["hd95"]) else "N/A"),
        **{f"student_{k}": v for k, v in ops_student.items() if k != "label"},
        "teacher_test":  teacher_test,
        "teacher_ops":   ops_teacher,
        "history_csv":   csv_path,
    }

    # Save per-seed metrics JSON
    os.makedirs(cfg.results_dir, exist_ok=True)
    per_seed_path = os.path.join(
        cfg.results_dir, f"{method}_seed{seed}_metrics.json")
    with open(per_seed_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Per-seed results → {per_seed_path}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-seed aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def aggregate_seed_results(per_seed: List[dict]) -> dict:
    """
    Compute mean ± std of test_dice, test_iou, test_hd95 across seeds.
    Operational metrics (params, GFLOPs, latency) taken from the last seed.
    """
    if not per_seed:
        return {}

    keys   = ["test_dice", "test_iou", "test_hd95"]
    values = {k: [_safe_float(r[k]) for r in per_seed] for k in keys}
    agg    = {"method": per_seed[0]["method"],
              "seeds_run": [r["seed"] for r in per_seed],
              "n_seeds": len(per_seed)}

    for k, vals in values.items():
        clean = [v for v in vals if not np.isnan(v)]
        if clean:
            agg[f"{k}_mean"] = round(float(np.mean(clean)), 4)
            agg[f"{k}_std"]  = round(float(np.std(clean)),  4)
        else:
            agg[f"{k}_mean"] = "N/A"
            agg[f"{k}_std"]  = "N/A"
        # Canonical scalar = mean (for backward-compatible table display)
        agg[k] = agg[f"{k}_mean"]

    # Carry operational metrics from the last available seed
    last = per_seed[-1]
    for k in ("student_params_M", "student_gflops",
              "student_latency_ms", "student_gpu_mem_mb"):
        agg[k] = last.get(k, "N/A")

    # Teacher metrics (only present for method="none")
    if last.get("teacher_test"):
        agg["teacher_test"] = last["teacher_test"]
    if last.get("teacher_ops"):
        agg["teacher_ops"] = last["teacher_ops"]

    return agg


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison table printer
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(results: list):
    """
    Print a formatted comparison table.  If results include mean±std fields
    (from multi-seed aggregation), both are shown.
    """
    multi = any("test_dice_std" in r for r in results)

    print("\n" + "=" * 105)
    print("  COMPARISON TABLE" + ("  (mean ± std across seeds)" if multi else ""))
    print("=" * 105)

    if multi:
        hdr = (f"{'Method':<12} {'Dice':>14} {'IoU':>14} {'HD95':>14} "
               f"{'Params(M)':>10} {'GFLOPs':>8} {'Lat(ms)':>9}")
    else:
        hdr = (f"{'Method':<12} {'Dice':>6} {'IoU':>6} {'HD95':>7} "
               f"{'Params(M)':>10} {'GFLOPs':>8} {'Lat(ms)':>9} {'GPU(MB)':>8}")
    print(hdr)
    print("-" * 105)

    for r in results:
        if multi:
            dice_s = (f"{r.get('test_dice_mean','N/A')}"
                      f"±{r.get('test_dice_std','N/A')}")
            iou_s  = (f"{r.get('test_iou_mean','N/A')}"
                      f"±{r.get('test_iou_std','N/A')}")
            hd95_s = (f"{r.get('test_hd95_mean','N/A')}"
                      f"±{r.get('test_hd95_std','N/A')}")
            print(f"{r['method']:<12} {dice_s:>14} {iou_s:>14} {hd95_s:>14} "
                  f"{str(r.get('student_params_M','N/A')):>10} "
                  f"{str(r.get('student_gflops','N/A')):>8} "
                  f"{str(r.get('student_latency_ms','N/A')):>9}")
        else:
            print(f"{r['method']:<12} "
                  f"{r.get('test_dice','N/A'):>6} "
                  f"{r.get('test_iou','N/A'):>6} "
                  f"{str(r.get('test_hd95','N/A')):>7} "
                  f"{str(r.get('student_params_M','N/A')):>10} "
                  f"{str(r.get('student_gflops','N/A')):>8} "
                  f"{r.get('student_latency_ms', 0):>9.1f} "
                  f"{str(r.get('student_gpu_mem_mb','N/A')):>8}")
    print("=" * 105)

    # Print teacher metrics if present
    none_result = next((r for r in results if r.get("method") == "none"), None)
    if none_result and none_result.get("teacher_test"):
        tt = none_result["teacher_test"]
        to = none_result.get("teacher_ops", {})
        print(f"\n  Teacher (ResNet50-UNet) — reference ceiling:")
        print(f"    Test dice={tt['dice']:.4f} | iou={tt['iou']:.4f} | "
              f"hd95={tt['hd95']:.2f} | "
              f"params={to.get('params_M','N/A')}M | "
              f"latency={to.get('latency_ms','N/A')}ms")

    out_path = os.path.join("./results", "comparison_results.json")
    os.makedirs("./results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark-only mode  (Fix 2 & 3: multi-seed aggregation + teacher metrics)
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark_only(cfg: Config):
    """
    Scan for all per-seed checkpoint files, evaluate each, aggregate across
    seeds, and print the final comparison table.

    Fix 2: automatically discovers best_{method}_seed*.pth and groups by method.
    Fix 3: evaluates the teacher (always deterministic; benchmarked once).
    """
    _, _, test_loader = build_dataloaders(cfg)

    agg_results = []

    for method in ALL_METHODS:
        ckpt_pattern = os.path.join(cfg.ckpt_dir, f"best_{method}_seed*.pth")
        ckpts = sorted(glob.glob(ckpt_pattern))

        # Backward compat: check for old-style checkpoint (no seed suffix)
        if not ckpts:
            legacy = os.path.join(cfg.ckpt_dir, f"best_{method}.pth")
            if os.path.exists(legacy):
                ckpts = [legacy]

        if not ckpts:
            print(f"  [SKIP] No checkpoints found for method={method}")
            continue

        per_seed = []
        ops_last = None

        for ckpt_path in ckpts:
            # Parse seed from filename (e.g. best_none_seed42.pth → 42)
            basename = os.path.basename(ckpt_path)
            try:
                seed_str = basename.replace(f"best_{method}_seed", "").replace(".pth", "")
                seed_val = int(seed_str)
            except ValueError:
                seed_val = 0   # legacy checkpoint without seed
                
            if method == "teacher":
                student = build_teacher(cfg, load_ckpt=False)
            else:
                student = build_student(cfg)
                
            ckpt    = torch.load(ckpt_path, map_location=cfg.device)
            student.load_state_dict(ckpt["student"])

            tm = evaluate(student, test_loader, cfg)

            if ops_last is None:
                ops_last = benchmark_model(student, cfg,
                                           label=f"student_{method}")

            per_seed.append({
                "method":            method,
                "seed":              seed_val,
                "test_dice":         round(tm["dice"], 4),
                "test_iou":          round(tm["iou"],  4),
                "test_hd95":         (round(tm["hd95"], 3)
                                      if not np.isnan(tm["hd95"]) else "N/A"),
                **{f"student_{k}": v for k, v in ops_last.items()
                   if k != "label"},
            })
            print(f"  [{method} seed={seed_val}] "
                  f"dice={tm['dice']:.4f} iou={tm['iou']:.4f} "
                  f"hd95={tm['hd95']:.2f}")

        agg = aggregate_seed_results(per_seed)

        # Fix 3: teacher metrics for 'none' method
        if method == "none":
            teacher = build_teacher(cfg, load_ckpt=True)
            print("  Evaluating teacher on test set …")
            tt = evaluate(teacher, test_loader, cfg)
            ops_t = benchmark_model(teacher, cfg, label="teacher")
            agg["teacher_test"] = {
                "dice": round(tt["dice"], 4),
                "iou":  round(tt["iou"],  4),
                "hd95": (round(tt["hd95"], 3)
                          if not np.isnan(tt["hd95"]) else "N/A"),
            }
            agg["teacher_ops"] = ops_t

        agg_results.append(agg)

    if agg_results:
        print_comparison_table(agg_results)
    else:
        print("  No checkpoints found.  Run training first.")

    return agg_results


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

ALL_METHODS = ["teacher", "none", "vanilla", "fitnets", "at", "ldl"]


def parse_args():
    p = argparse.ArgumentParser(description="LDL KD Comparison Runner")
    p.add_argument("--method", default="ldl",
                   choices=ALL_METHODS + ["all"],
                   help="KD method to run (or 'all' for full comparison)")
    p.add_argument("--data_root",  default="./BUSI")
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lambda_kd",  type=float, default=None)
    p.add_argument("--benchmark_only", action="store_true",
                   help="Aggregate all saved checkpoints and print results")
    # Fix 2: single seed per invocation; call script multiple times for multi-seed
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for this training run (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()

    cfg           = Config()
    cfg.data_root = args.data_root
    if args.epochs     is not None: cfg.epochs     = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.lambda_kd  is not None: cfg.lambda_kd  = args.lambda_kd
    cfg.seed = args.seed

    print(f"\n  Device : {cfg.device}")
    print(f"  AMP    : {cfg.amp}")
    print(f"  Epochs : {cfg.epochs}")
    print(f"  Batch  : {cfg.batch_size}")
    print(f"  Seed   : {cfg.seed}")
    print(f"  Teacher: {cfg.teacher_encoder}")
    print(f"  Student: {cfg.student_encoder}")

    if args.benchmark_only:
        run_benchmark_only(cfg)
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