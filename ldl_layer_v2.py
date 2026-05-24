"""
ldl_layer.py — Laguerre Distillation Layer (LDL)  [Revised v2]

Implements Algorithms 1–3 from the proposal, grounded in:
  Chung, Han, Li, Li. "Unbalanced Optimal Total Variation Transport." NeurIPS 2025.

Revision log (addressing reviewer feedback):
─────────────────────────────────────────────────────────────────────────────
Issue 1 [CRITICAL] — Radial cost assumption (Definition 3.2, base paper)
    OLD: c(p, y_i) = α‖p_sp − y_sp‖² + (1−α)(1 − cos⟨p_feat, y_feat⟩)
         The cosine term depends on the *content* of φ(p), not just d(p, y_i),
         so the combined cost is NOT radial and Theorems 3.3, Prop 3.1–3.2
         do not apply.
    FIX: Embed both sides onto the unit (D−1)-sphere before computing L2²:
         c(p̂, ŷ_i) = α‖p_sp − y_sp‖² + (1−α)‖p̂_feat − ŷ_feat‖²
         where  p̂_feat = normalize(φ(p)),  ŷ_feat = normalize(y_feat).
         The augmented point lives in ℝ² × 𝕊^{D−1} ⊂ ℝ^{D+2} and the cost
         is c = g(d) with g(t) = t², d = weighted Euclidean on ℝ^{D+2}.
         This satisfies Definition 3.2 (strictly increasing g, radial in a
         product metric space). Note: ‖n_a − n_b‖² = 2(1 − cos⟨n_a, n_b⟩),
         so the change is a constant rescaling of (1−α), not a loss of
         semantic discriminability.

Issue 2 [SIGNIFICANT] — Prop. 3.2 operates at w = 0 vs. operational w*
    OLD: Gradient claimed to follow Prop. 3.2 exactly, but Prop. 3.2 requires
         masses set at w = 0 cells (C_i(0)), while training uses w* cells.
    FIX: The code retains the autograd graph through V = kC − w*ᵀ and applies
         a stop-gradient (straight-through estimator) only on the cell-
         assignment argmin, so gradients actually flow through C_i(w*) — the
         dual-weight-adjusted cells. This is a straight-through approximation
         of the exact gradient at w*, acknowledged explicitly in the docstring.
         It is strictly more accurate than using C_i(0) when w* ≠ 0 (which
         occurs exactly in the high-imbalance regime the method targets).

Issue 3 [SIGNIFICANT] — Straight-through estimator not disclosed
    FIX: Fully documented in forward() and in the module docstring. Language
         in the proposal ("exact closed-form gradient") updated to
         ("straight-through approximation of the Proposition 3.2 gradient").

Issue 4 [SIGNIFICANT] — ℓ_anc-reg: min_j is non-differentiable
    OLD: l_anc_reg = (1/M) Σ_i  min_j ‖y_feat_i − f̂_S_{j,feat}‖²
         The hard min has sub-gradients only at the attaining index; PyTorch's
         autograd returns zero gradient elsewhere, creating dead anchors.
    FIX: Replace with a soft-min (negative log-sum-exp of negative distances):
         softmin_τ(d) = −τ · log Σ_j exp(−d_{ij}/τ)
         This is smooth, lower-bounds the hard min (softmin → min as τ→0),
         and provides dense gradients to all anchor-feature pairs.
         Temperature τ = 0.1 is registered as a hyperparameter.

Issue 5 [SIGNIFICANT] — k-means++ runs BEFORE ψ_T warm-up
    OLD: warmup_anchors() called on random ψ_T embeddings; after 500-step
         warm-up ψ_T changes, leaving anchors misaligned with the frozen
         embedding space.
    FIX: warmup_anchors() must be called AFTER ψ_T is frozen (i.e., after
         the warm-up phase). A runtime guard (_psi_T_frozen flag) enforces
         this and raises a descriptive error if violated.
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class LaguerreDistillationLayer(nn.Module):
    """
    Plug-and-play KD module for dense CNN students (Algorithm 3, revised).

    Transport cost — RADIAL in ℝ^{D+2} with weighted Euclidean metric:
        c(p̂, ŷ_i) = α ‖p_sp − y_sp‖²  +  (1−α) ‖norm(p_feat) − norm(y_feat)‖²
    where norm(·) is L2 normalisation onto 𝕊^{D−1}. This satisfies
    Definition 3.2 of Chung et al. (radial cost), restoring the validity of
    Theorem 3.3 (dual tessellation) and Propositions 3.1–3.2.

    Location gradient — straight-through approximation:
        Gradients flow through the *cost values* in C_i(w*)-partitioned cells
        (i.e. the dual-weight-adjusted partition), with a stop-gradient applied
        only to the argmin cell-assignment indicator. This is a straight-through
        estimator (STE) of the Proposition 3.2 gradient; it is NOT the exact
        closed-form gradient at w = 0, but is more accurate in the unbalanced
        regime where w* ≠ 0 (because the integration region tracks C_i(w*),
        not C_i(0)).

    Anchor regulariser — smooth soft-min:
        ℓ_anc-reg uses negative log-sum-exp (soft-min with temperature τ)
        instead of hard argmin, giving dense, everywhere-defined gradients.

    Initialisation order:
        1. Instantiate LDL.
        2. Warm up ψ_T for ~500 gradient steps and FREEZE it.
        3. Call warmup_anchors() — this requires _psi_T_frozen = True.
        4. Begin joint LDL + student training.

    Args:
        teacher_channels : channel width of teacher distillation feature map
        student_channels : channel width of student distillation feature map
        embed_dim        : shared projection dimension D
        num_anchors      : number of Laguerre anchors M
        alpha            : spatial vs. semantic cost weight ∈ (0,1)
        k                : transport-cost scale factor
        a1, b1           : newsvendor TV parameters, supply (teacher) side
        a2, b2           : newsvendor TV parameters, demand (student) side
        T_w              : inner dual-solver steps per batch
        eta0, beta       : decaying step-size η_t = η0·(1+t)^{−β}
        anc_reg          : anchor regularisation weight γ
        anc_reg_tau      : soft-min temperature for ℓ_anc-reg (Issue 4 fix)
        device           : torch device string
    """

    def __init__(
        self,
        teacher_channels : int,
        student_channels : int,
        embed_dim        : int   = 128,
        num_anchors      : int   = 32,
        alpha            : float = 0.5,
        k                : float = 1.0,
        a1               : float = 0.30,
        b1               : float = 0.10,
        a2               : float = 1.00,
        b2               : float = 0.20,
        T_w              : int   = 15,
        eta0             : float = 0.05,
        beta             : float = 0.60,
        anc_reg          : float = 0.01,
        anc_reg_tau      : float = 0.10,   # Issue 4: soft-min temperature
        device           : str   = "cpu",
    ):
        super().__init__()
        self.M           = num_anchors
        self.D           = embed_dim
        self.alpha       = alpha
        self.k           = k
        self.a1, self.b1 = a1, b1
        self.a2, self.b2 = a2, b2
        self.T_w         = T_w
        self.eta0        = eta0
        self.beta        = beta
        self.anc_reg     = anc_reg
        self.anc_reg_tau = anc_reg_tau

        # ── Projection adapters (1×1 conv, no bias) ──────────────────────
        self.psi_T = nn.Conv2d(teacher_channels, embed_dim, 1, bias=False)
        self.psi_S = nn.Conv2d(student_channels, embed_dim, 1, bias=False)
        nn.init.kaiming_normal_(self.psi_T.weight)
        nn.init.kaiming_normal_(self.psi_S.weight)

        # ── Learnable anchor prototypes ŷ_i ∈ ℝ^{D+2} ───────────────────
        # Semantic component (first D dims) stored un-normalised;
        # normalisation applied inside _cost_matrix to satisfy the radial
        # condition (Issue 1 fix). Spatial component (last 2 dims) is
        # initialised small and learned freely.
        self.anchors = nn.Parameter(
            torch.randn(num_anchors, embed_dim + 2) * 0.02
        )

        # ── Non-learnable state buffers ───────────────────────────────────
        self.register_buffer("w",      torch.zeros(num_anchors))
        self.register_buffer("masses", torch.ones(num_anchors) / num_anchors)
        self.register_buffer("_step",  torch.zeros(1, dtype=torch.long))

        # Issue 5 fix: guard that enforces correct initialisation order.
        # Set to True only after ψ_T warm-up is complete and psi_T.requires_grad
        # has been set to False by the caller.
        self._psi_T_frozen: bool = False

    # ====================================================================
    # I1 / I2 — TV penalty functions  (Chung et al., Eq. 3)
    # ====================================================================
    def _I1(self, xi: torch.Tensor) -> torch.Tensor:
        """I1(ξ) = clamp(ξ, −∞, a1) restricted to active region."""
        return torch.where(xi > self.a1,
                           xi.new_full((), self.a1).expand_as(xi), xi)

    def _I2(self, wi: torch.Tensor) -> torch.Tensor:
        """I2(w_i) = clamp(w_i, −b2, a2)."""
        return wi.clamp(min=-self.b2, max=self.a2)

    # ====================================================================
    # Cost matrix  C ∈ ℝ^{B × N × M}   [Issue 1 — radial cost]
    # ====================================================================
    def _cost_matrix(
        self,
        F_hat_T: torch.Tensor,   # (B, N, D+2)
    ) -> torch.Tensor:           # (B, N, M)
        """
        Radial cost on the product space ℝ² × 𝕊^{D−1}:

            c(p̂, ŷ_i) = α · ‖p_sp − y_sp‖²
                       + (1−α) · ‖normalize(p_feat) − normalize(y_feat)‖²

        This satisfies Definition 3.2 of Chung et al. because:
          • d(p̂, ŷ_i)² = α‖Δ_sp‖² + (1−α)‖Δ_feat‖² is a weighted Euclidean
            distance on ℝ^{D+2} (a valid metric on ℝ² × 𝕊^{D−1}).
          • c = g(d) with g(t) = t², which is continuous and strictly
            increasing on [0, ∞). ✓ Radial.

        Note: ‖n_a − n_b‖² = 2(1 − ⟨n_a, n_b⟩) for unit vectors, so the
        semantic term is equivalent to twice the cosine distance. The factor
        of 2 is absorbed into the (1−α) coefficient without loss of generality.
        """
        p_sp   = F_hat_T[:, :, self.D:]        # (B, N, 2)
        p_feat = F_hat_T[:, :, :self.D]        # (B, N, D)

        y_sp   = self.anchors[:, self.D:]      # (M, 2)
        y_feat = self.anchors[:, :self.D]      # (M, D)

        # ── Spatial L2²  (already radial) ───────────────────────────────
        diff    = p_sp.unsqueeze(2) - y_sp.unsqueeze(0).unsqueeze(0)  # (B,N,M,2)
        sp_cost = (diff ** 2).sum(dim=-1)       # (B, N, M)

        # ── Semantic L2² on unit sphere (Issue 1 fix) ────────────────────
        # Both sides normalised → cost is a squared Euclidean distance in
        # ℝ^D, restoring radial structure.  Gradient flows through p_feat
        # (anchors) and is well-defined at all non-antipodal pairs.
        p_norm  = F.normalize(p_feat, dim=-1)   # (B, N, D)
        y_norm  = F.normalize(y_feat, dim=-1)   # (M, D)
        # ‖p_n − y_n‖² = 2 − 2⟨p_n, y_n⟩; computed via einsum for efficiency
        dot     = torch.einsum("bnd,md->bnm", p_norm, y_norm)  # (B, N, M)
        sem_cost = 2.0 - 2.0 * dot                              # (B, N, M)
        # sem_cost ∈ [0, 4] for unit vectors; scale to [0,1] by dividing by 4
        # so the α/(1-α) balance is numerically comparable to sp_cost ∈ [0,2].
        sem_cost = sem_cost / 4.0

        return self.alpha * sp_cost + (1.0 - self.alpha) * sem_cost  # (B,N,M)

    # ====================================================================
    # Algorithm 1 — DualSolve  (no_grad, vectorised)
    # ====================================================================
    @torch.no_grad()
    def _dual_solve(self, C_mean: torch.Tensor) -> torch.Tensor:
        """
        Projected sub-gradient ascent on G(w) (Theorem 3.3, Prop. 3.1).

        Sub-gradient at w_i  (Proposition 3.1 discretised):
            g_i = −µ_i(C_i(w) ∩ A_i)  +  m_i  ·  𝟙[w_i < a2]
        where µ_i is the empirical mass of the *active* part of cell i.

        Args:
            C_mean : (N, M)  cost matrix averaged over the batch (detached)
        Returns:
            w*     : (M,)    updated dual weights
        """
        w    = self.w.clone()
        N, M = C_mean.shape

        for t in range(1, self.T_w + 1):
            eta_t = self.eta0 * float(1.0 + t) ** (-self.beta)

            V             = self.k * C_mean - w.unsqueeze(0)   # (N, M)
            v_min, sigma  = V.min(dim=1)                        # (N,), (N,)

            cell_oh       = torch.zeros(N, M, device=w.device)
            cell_oh.scatter_(1, sigma.unsqueeze(1), 1.0)        # (N, M)

            in_range      = ((v_min >= -self.b1) & (v_min <= self.a1)).float()
            active        = cell_oh * in_range.unsqueeze(1)     # (N, M)
            mu            = active.mean(dim=0)                  # (M,)

            # Sub-gradient per Proposition 3.1 (three-case formula)
            below_a2 = (w <  self.a2 - 1e-8).float()
            at_a2    = ((w - self.a2).abs() < 1e-8).float()
            g        = -mu + self.masses * (below_a2 + 0.5 * at_a2)

            w = (w + eta_t * g).clamp(min=-self.b2)

        return w

    # ====================================================================
    # Mass update — once per epoch  (Theorem 3.5)
    # ====================================================================
    @torch.no_grad()
    def update_masses(self, C_mean: torch.Tensor):
        """
        m_i ← |C_i(w) ∩ Ω| / |Ω|   (optimal mass given current locations).
        C_mean: (N, M) cost matrix averaged over the training set (detached).
        """
        V     = self.k * C_mean - self.w.unsqueeze(0)
        sigma = V.argmin(dim=1)
        for i in range(self.M):
            self.masses[i] = (sigma == i).float().mean().clamp(min=1e-4)

    # ====================================================================
    # Anchor warmup — k-means++ AFTER ψ_T is frozen  [Issue 5]
    # ====================================================================
    @torch.no_grad()
    def freeze_psi_T(self):
        """
        Freeze the teacher adapter ψ_T and mark it as ready for k-means++.
        Call this after the ψ_T warm-up phase (~500 gradient steps).
        """
        for p in self.psi_T.parameters():
            p.requires_grad_(False)
        self._psi_T_frozen = True
        print("  [LDL] ψ_T frozen. Call warmup_anchors() now.")

    @torch.no_grad()
    def warmup_anchors(self, feat_T_batches: List[torch.Tensor]):
        """
        Initialise anchor semantic components via k-means++ on ψ_T-projected
        teacher features.  Must be called AFTER freeze_psi_T() so that anchors
        are initialised in the *same* embedding space that will be used during
        training. (Issue 5 fix.)

        Args:
            feat_T_batches : list of (B, D, H, W) tensors — psi_T-projected
                             teacher features collected after ψ_T is frozen.
        """
        if not self._psi_T_frozen:
            raise RuntimeError(
                "warmup_anchors() must be called AFTER freeze_psi_T(). "
                "Run the ψ_T warm-up phase first (~500 gradient steps), "
                "then call ldl.freeze_psi_T(), then collect projected features "
                "and call ldl.warmup_anchors(feat_T_batches)."
            )

        pts = []
        for f in feat_T_batches:
            B, D, H, W = f.shape
            # L2-normalise to match the embedding space used in _cost_matrix
            f_norm = F.normalize(f, dim=1)          # normalise along channel
            pts.append(f_norm.permute(0, 2, 3, 1).reshape(-1, D).cpu())
        pts = torch.cat(pts, dim=0)                  # (N_total, D)

        # k-means++ initialisation (O(M · N_total · D))
        centers = [pts[torch.randint(len(pts), (1,)).item()]]
        for _ in range(self.M - 1):
            dists = torch.cdist(pts.float(),
                                torch.stack(centers).float()).min(dim=1).values
            probs = dists ** 2 / (dists ** 2 + 1e-12).sum()
            idx   = torch.multinomial(probs, 1).item()
            centers.append(pts[idx])
        centers = torch.stack(centers)               # (M, D)

        # Store as un-normalised; _cost_matrix will normalise at each step.
        # Spatial dims initialised to [0.5, 0.5] (feature-map centre).
        self.anchors.data[:, :self.D] = centers.to(self.anchors.device)
        self.anchors.data[:, self.D:] = 0.5          # neutral spatial init
        print(f"  [LDL] Anchors initialised via k-means++ on "
              f"{len(pts):,} positions → {self.M} anchors  "
              f"(called after ψ_T frozen ✓)")

    # ====================================================================
    # Forward pass — Algorithm 3 (dense CNN / prototype-based)
    # ====================================================================
    def forward(
        self,
        feat_T: torch.Tensor,   # (B, C_T, H, W)  teacher feature (frozen)
        feat_S: torch.Tensor,   # (B, C_S, H, W)  student feature
    ) -> torch.Tensor:
        """
        Returns the scalar LDL loss  ℒ_LDL  (Algorithm 3).

        Gradient accounting [Issues 2 & 3]:
        ─────────────────────────────────────────────────────────────────
        • Dual weights w are updated by DualSolve (no_grad, in-place).
        • The loss is re-evaluated at the converged w* with grad retained
          through the cost computation (16) — so gradients flow through
          C_i(w*)-partitioned cells, not C_i(0).
        • The argmin cell-assignment (σ) is treated as a straight-through
          estimator (stop-gradient), following the convention of VQ-VAE
          and related discrete latent variable models.
        • This is a STRAIGHT-THROUGH APPROXIMATION of the exact gradient,
          not an exact closed-form expression. The approximation error is
          bounded by the boundary mass µ₁(∂C_i(w*)) which is zero µ₁-a.e.
          when c is radial and µ₁ is absolutely continuous (Lemma 3.8 of
          Bourne et al. [15], which now applies because c is radial ✓).
        ─────────────────────────────────────────────────────────────────
        """
        B, _, H, W = feat_T.shape
        N = H * W
        self._step += 1

        # ── Step 1: project features ──────────────────────────────────────
        ft = self.psi_T(feat_T)   # (B, D, H, W)  [ψ_T frozen after warmup]
        fs = self.psi_S(feat_S)   # (B, D, H, W)

        # ── Step 2: spatial grid  P ∈ [0,1]² ─────────────────────────────
        grid_h = torch.linspace(0, 1, H, device=feat_T.device)
        grid_w = torch.linspace(0, 1, W, device=feat_T.device)
        gy, gx = torch.meshgrid(grid_h, grid_w, indexing="ij")
        P = torch.stack([gy, gx], dim=-1).reshape(N, 2)          # (N, 2)
        P = P.unsqueeze(0).expand(B, -1, -1)                     # (B, N, 2)

        # ── Step 3: augmented teacher feature  F̂_T = [ft_flat | P] ──────
        ft_flat = ft.permute(0, 2, 3, 1).reshape(B, N, self.D)   # (B, N, D)
        F_hat_T = torch.cat([ft_flat, P], dim=-1)                 # (B, N, D+2)

        # ── Step 4: cost matrix  C ∈ ℝ^{B×N×M} (radial cost, Issue 1) ──
        C = self._cost_matrix(F_hat_T)                            # (B, N, M)

        # ── Step 5: DualSolve — inner loop, no grad ───────────────────────
        C_mean_detach = C.detach().mean(0)   # (N, M)
        w_star = self._dual_solve(C_mean_detach)
        self.w.copy_(w_star)

        # ── Step 6: LDL loss with straight-through gradient (Issues 2 & 3) ─
        #    V computed at w* → gradients track C_i(w*) cells.
        #    σ = argmin(V) treated as stop-grad (STE).
        V      = self.k * C - self.w.detach().unsqueeze(0).unsqueeze(0)  # (B,N,M)
        v_min  = V.min(dim=2).values                                      # (B, N)
        c_w    = v_min.clamp(min=-self.b1)   # c^w = (−b1) ∨ min_i{kc − w_i}

        residual_mask = (v_min > self.a1)    # (B, N) — Residual Set ℛ
        active_mask   = ~residual_mask

        # Term 1: active transport cost  (1/|Ω\ℛ|) Σ_{p∉ℛ} I₁(c^w_p)
        I1_vals      = self._I1(c_w)
        n_active     = active_mask.float().sum() + 1e-6
        l_transport  = (I1_vals * active_mask.float()).sum() / n_active

        # Term 2: residual penalty  a1 · |ℛ| / |Ω|
        l_residual   = self.a1 * residual_mask.float().mean()

        # Term 3: anchor alignment  Σ_i I₂(w*_i) · m_i
        l_anchor     = (self._I2(self.w.detach()) * self.masses.detach()).sum()

        # ── Step 7: anchor regularisation — smooth soft-min  (Issue 4) ────
        #    HARD MIN (old): l_anc_reg = min_j ‖y_i − f̂_S_j‖²
        #      → non-differentiable at the attaining index j*;
        #        PyTorch returns 0 gradient for j ≠ j*, creating dead anchors.
        #    SOFT-MIN (new): softmin_τ(d_{ij}) = −τ · log Σ_j exp(−d_{ij}/τ)
        #      → everywhere differentiable; recovers hard min as τ → 0⁺;
        #        gives dense gradients to all anchor-student-feature pairs.
        fs_flat = fs.permute(0, 2, 3, 1).reshape(B, N, self.D)  # (B, N, D)
        F_hat_S = torch.cat([fs_flat, P.detach()], dim=-1)       # (B, N, D+2)

        # d_anc[b, i, j] = ‖ŷ_i − f̂_S_{b,j}‖²   (B, M, N)
        d_anc   = torch.cdist(
            self.anchors.unsqueeze(0).expand(B, -1, -1),   # (B, M, D+2)
            F_hat_S,                                        # (B, N, D+2)
        )                                                   # (B, M, N)

        # Soft-min over j (student positions) for each anchor i
        # = −τ · log Σ_j exp(−d_{ij} / τ)
        tau          = self.anc_reg_tau
        l_anc_reg   = (-tau * torch.logsumexp(
            -d_anc / tau, dim=2               # logsumexp over N positions
        )).mean()                              # mean over B, M

        # ── Store diagnostic info ─────────────────────────────────────────
        with torch.no_grad():
            self._last_residual_frac = residual_mask.float().mean().item()

        loss = l_transport + l_residual + l_anchor + self.anc_reg * l_anc_reg
        return loss

    # ── Diagnostic properties ──────────────────────────────────────────────
    @property
    def residual_fraction(self) -> float:
        """Fraction of last-batch positions in residual set ℛ."""
        return getattr(self, "_last_residual_frac", float("nan"))

    def extra_repr(self) -> str:
        return (f"M={self.M}, D={self.D}, α={self.alpha}, k={self.k}, "
                f"a1={self.a1}, b1={self.b1}, a2={self.a2}, b2={self.b2}, "
                f"T_w={self.T_w}, γ={self.anc_reg}, τ={self.anc_reg_tau}")
