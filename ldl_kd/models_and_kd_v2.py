"""
models_and_kd.py — Model Factory + All KD Baseline Losses
Architectures:
  Teacher : ResNet34-UNet   (~24M params, ResNet34 encoder)
  Student : MobileV2-UNet   (~4M  params, MobileNetV2 encoder)  ← edge-deployable
KD Baselines:
  VanillaKD  — Hinton et al. (2015), soft logit matching
  FitNets    — Romero et al. (2014), intermediate feature L2
  AT         — Zagoruyko & Komodakis (2017), attention map matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Dict, Optional, Tuple


# ================================================================
# MODEL FACTORY
# ================================================================
def build_teacher(cfg) -> nn.Module:
    """ResNet34-UNet: strong teacher, ~24M params."""
    model = smp.Unet(
        encoder_name=cfg.teacher_encoder,
        encoder_weights=cfg.encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,          # raw logits; we apply sigmoid in loss
        decoder_channels=(256, 128, 64, 32, 16),
    )
    return model.to(cfg.device)


def build_student(cfg) -> nn.Module:
    """MobileNetV2-UNet: compact student, ~4M params, INT8-quantizable."""
    model = smp.Unet(
        encoder_name=cfg.student_encoder,
        encoder_weights=cfg.encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
        decoder_channels=(128, 64, 32, 16, 8),  # lighter decoder for edge
    )
    return model.to(cfg.device)


def get_encoder_feat(model: nn.Module, x: torch.Tensor,
                     feat_idx: int = -2) -> Tuple[torch.Tensor, torch.Tensor]:
    features = model.encoder(x)      # list of tensors
    feat_map = features[feat_idx]    # (B, C, H', W')  ← distillation feature
    
    # Xử lý tương thích đa phiên bản cho SMP
    try:
        # Dành cho SMP phiên bản cũ
        decoder_out = model.decoder(*features)
    except TypeError:
        # Dành cho SMP phiên bản mới (như trên Kaggle hiện tại)
        decoder_out = model.decoder(features)
        
    logit = model.segmentation_head(decoder_out)   # (B, 1, H, W)
    return logit, feat_map

# ================================================================
# SEGMENTATION LOSS (used for all methods)
# ================================================================
class SegLoss(nn.Module):
    """Dice + BCE combined loss (standard for medical segmentation)."""
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred_logit: torch.Tensor,
                  target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pred = torch.sigmoid(pred_logit)
        inter = (pred * target).sum(dim=(1, 2, 3))
        denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        return 1.0 - (2 * inter + eps) / (denom + eps)

    def forward(self, pred_logit: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        bce  = self.bce(pred_logit, target)
        dice = self.dice_loss(pred_logit, target).mean()
        return self.bce_weight * bce + self.dice_weight * dice


# ================================================================
# KD BASELINE 1: VANILLA KD (Hinton et al. 2015)
# ================================================================
class VanillaKDLoss(nn.Module):
    """
    Soft-logit knowledge distillation.
    KL( σ(teacher/T) ‖ σ(student/T) ) scaled by T².
    Applied to segmentation maps after flattening spatial dims.
    """
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_logit: torch.Tensor,
                teacher_logit: torch.Tensor) -> torch.Tensor:
        B = student_logit.shape[0]
        s = (student_logit / self.T).reshape(B, -1)
        t = (teacher_logit / self.T).reshape(B, -1).detach()

        # Binary segmentation → sigmoid soft labels
        p_t = torch.sigmoid(t)   # teacher soft probability
        p_s = torch.sigmoid(s)   # student soft probability

        # KL divergence for binary case: sum over pixels
        loss = F.binary_cross_entropy(p_s, p_t, reduction="mean")
        return loss * (self.T ** 2)


# ================================================================
# KD BASELINE 2: FitNets (Romero et al. 2014)
# ================================================================
class FitNetsLoss(nn.Module):
    """
    L2 feature matching with a learned linear projector.
    Projector maps student channels → teacher channels.
    """
    def __init__(self, student_channels: int, teacher_channels: int,
                 device: str = "cpu"):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, 1, bias=False),
            nn.BatchNorm2d(teacher_channels),
        ).to(device)

    def forward(self, feat_s: torch.Tensor,
                feat_t: torch.Tensor) -> torch.Tensor:
        feat_t = feat_t.detach()
        # Align spatial size if needed
        if feat_s.shape[-2:] != feat_t.shape[-2:]:
            feat_s = F.interpolate(feat_s, size=feat_t.shape[-2:],
                                   mode="bilinear", align_corners=False)
        feat_s_proj = self.projector(feat_s)
        return F.mse_loss(feat_s_proj, feat_t)


# ================================================================
# KD BASELINE 3: Attention Transfer (Zagoruyko & Komodakis, 2017)
# ================================================================
class ATLoss(nn.Module):
    """
    Attention map matching.
    Attention map A(F) = L2-normalised sum of squared activations.
    Loss = ‖ A(teacher) - A(student) ‖₂
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def attention_map(feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, H, W) → attention: (B, 1, H, W), L2-normalised."""
        a = feat.pow(2).sum(dim=1, keepdim=True)      # (B, 1, H, W)
        a_flat = a.reshape(a.shape[0], -1)
        a_norm = a_flat / (a_flat.norm(dim=1, keepdim=True) + 1e-8)
        return a_norm.reshape_as(a)

    def forward(self, feat_s: torch.Tensor,
                feat_t: torch.Tensor) -> torch.Tensor:
        feat_t = feat_t.detach()
        if feat_s.shape[-2:] != feat_t.shape[-2:]:
            feat_s = F.interpolate(feat_s, size=feat_t.shape[-2:],
                                   mode="bilinear", align_corners=False)
        at_t = self.attention_map(feat_t)
        at_s = self.attention_map(feat_s)
        return F.mse_loss(at_s, at_t)


# ================================================================
# KD METHOD REGISTRY
# ================================================================
def build_kd_loss(method: str, cfg, teacher_ch: int,
                  student_ch: int) -> Optional[nn.Module]:
    """
    Return the KD loss module for the given method name.
    Returns None for 'none' (student trained standalone).
    """
    if method == "none":
        return None
    elif method == "vanilla":
        return VanillaKDLoss(cfg.temperature).to(cfg.device)
    elif method == "fitnets":
        return FitNetsLoss(student_ch, teacher_ch,
                           cfg.device).to(cfg.device)
    elif method == "at":
        return ATLoss().to(cfg.device)
    elif method == "ldl":
        # Revised LDL module (ldl_layer_v2) with five reviewer fixes applied.
        from ldl_layer_v2 import LaguerreDistillationLayer
        return LaguerreDistillationLayer(
            teacher_channels=teacher_ch,
            student_channels=student_ch,
            embed_dim=cfg.ldl_embed_dim,
            num_anchors=cfg.ldl_num_anchors,
            alpha=cfg.ldl_alpha,
            k=cfg.ldl_k,
            a1=cfg.ldl_a1, b1=cfg.ldl_b1,
            a2=cfg.ldl_a2, b2=cfg.ldl_b2,
            T_w=cfg.ldl_T_w,
            eta0=cfg.ldl_eta0,
            beta=cfg.ldl_beta,
            anc_reg=cfg.ldl_anc_reg,
            anc_reg_tau=cfg.ldl_anc_reg_tau,   # Issue 4 fix: soft-min τ
            device=cfg.device,
        ).to(cfg.device)
    else:
        raise ValueError(f"Unknown KD method: {method!r}")
