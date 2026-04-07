"""
EIT Dataset – synthetic data generation via a linearised forward model.

Physics background
------------------
For a homogeneous 2-D circular disc of radius R and unit conductivity, the
electric potential due to unit current injected at angle α (source) and
extracted at angle β (sink) is given by the Fourier-series solution:

    φ(r, θ) = (1/π) Σ_{n=1}^{N} (1/n) (r/R)^n [cos n(θ−α) − cos n(θ−β)]

Its Cartesian gradient, evaluated at interior pixel centres, feeds the
Geselowitz sensitivity theorem (Born approximation):

    J[L, j] = −(∇φ_drive · ∇φ_meas)_j × pixel_area

ΔV = J @ Δσ  relates the 224-D measurement vector to pixel conductivity
changes Δσ (binary: 1 inside anomaly, 0 outside).

Electrode / measurement protocol
----------------------------------
* 16 electrodes at angles 2πk/16, k = 0 … 15.
* Adjacent drive pair k  → electrodes (k, k+1 mod 16).
* Adjacent measure pair m → electrodes (m, m+1 mod 16).
* 32 self-measurements excluded: m == k  OR  m == (k+1) mod 16.
  → 16 drives × 14 measures = 224 independent measurements.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Forward model
# ──────────────────────────────────────────────────────────────────────────────

class EITForwardModel:
    """
    Analytical EIT forward model for a 2-D circular homogeneous tank.

    Precomputes the Geselowitz sensitivity matrix J ∈ ℝ^{224 × img²} once
    using the Fourier-series potential gradient.  Subsequent data generation
    is a fast batched matrix multiply: ΔV = Δσ @ J.T.

    Parameters
    ----------
    config : dict
        Keys consumed: tank_radius_mm, n_electrodes, n_harmonics, image_size,
        anomaly_min_radius_mm, anomaly_max_radius_mm, noise_std.
    """

    def __init__(self, config: dict) -> None:
        self.R       = float(config["tank_radius_mm"])
        self.n_el    = int(config["n_electrodes"])
        self.N_h     = int(config["n_harmonics"])
        self.img_sz  = int(config["image_size"])
        self.r_min   = float(config["anomaly_min_radius_mm"])
        self.r_max   = float(config["anomaly_max_radius_mm"])
        self.noise_std = float(config["noise_std"])

        self._build_grid()
        self._build_electrodes()
        self._build_sensitivity()

    # ── Grid ─────────────────────────────────────────────────────────────────

    def _build_grid(self) -> None:
        """
        32×32 pixel grid with centres uniformly mapped to (−R, R) mm.

        Convention: row 0 → largest y (top of image), row 31 → smallest y,
        matching standard image-display orientation while keeping the physical
        coordinate system right-handed.
        """
        n = self.img_sz
        # Pixel centres in mm along one axis
        coords = (np.arange(n) + 0.5) * (2.0 * self.R / n) - self.R  # (n,)
        # xx: x increases left→right; yy: y increases bottom→top (flip rows)
        xx, yy = np.meshgrid(coords, coords[::-1])   # each (n, n)
        self.xx = xx
        self.yy = yy
        self.tank_mask  = (xx**2 + yy**2) <= self.R**2  # (n, n) bool
        self.pixel_area = (2.0 * self.R / n) ** 2        # mm²

        # Flat views for vectorised sensitivity computation
        self._r_flat  = np.sqrt(xx.ravel()**2 + yy.ravel()**2)   # (n²,)
        self._th_flat = np.arctan2(yy.ravel(), xx.ravel())        # (n²,)

    # ── Electrodes ────────────────────────────────────────────────────────────

    def _build_electrodes(self) -> None:
        self.el_angles = 2.0 * np.pi * np.arange(self.n_el) / self.n_el

    # ── Fourier-series gradient ───────────────────────────────────────────────

    def _grad_phi(self, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Cartesian gradient ∇φ at every pixel centre for a drive pair (α, β).

        Derivation:
            ∂φ/∂r         = (1/πR) Σ_{n=1}^N (r/R)^{n−1} [cos n(θ−α) − cos n(θ−β)]
            (1/r) ∂φ/∂θ  = (1/πR) Σ_{n=1}^N (r/R)^{n−1} [sin n(θ−β) − sin n(θ−α)]

            ∂φ/∂x = cos θ · ∂φ/∂r − sin θ · (1/r)∂φ/∂θ
            ∂φ/∂y = sin θ · ∂φ/∂r + cos θ · (1/r)∂φ/∂θ

        Returns
        -------
        grad_x, grad_y : np.ndarray, shape (image_size²,)
        """
        r  = self._r_flat    # (M,)
        th = self._th_flat   # (M,)
        ns = np.arange(1, self.N_h + 1)  # (N_h,)

        # (r/R)^{n−1}: broadcast (M, N_h)
        r_pow = (r[:, None] / self.R) ** (ns[None, :] - 1)

        # Angular argument matrices (M, N_h)
        nda = ns[None, :] * (th[:, None] - alpha)
        ndb = ns[None, :] * (th[:, None] - beta)

        # Summed series — shape (M,)
        sum_A = (r_pow * (np.cos(nda) - np.cos(ndb))).sum(axis=1)  # for ∂/∂r
        sum_B = (r_pow * (np.sin(ndb) - np.sin(nda))).sum(axis=1)  # for (1/r)∂/∂θ

        scale  = 1.0 / (np.pi * self.R)
        cos_th = np.where(r > 0.0, np.cos(th), 0.0)
        sin_th = np.where(r > 0.0, np.sin(th), 0.0)

        grad_x = scale * (cos_th * sum_A - sin_th * sum_B)
        grad_y = scale * (sin_th * sum_A + cos_th * sum_B)
        return grad_x, grad_y

    # ── Sensitivity matrix ────────────────────────────────────────────────────

    def _build_sensitivity(self) -> None:
        """
        Assemble J ∈ ℝ^{224 × img²}.

            J[L, j] = −(∇φ_drive · ∇φ_meas)_j × pixel_area

        Row order: drive k = 0…15, then measure m ≠ k and m ≠ (k+1) mod n_el.
        """
        # Precompute gradients for all 16 adjacent drives
        grads: list[tuple[np.ndarray, np.ndarray]] = []
        for k in range(self.n_el):
            alpha = self.el_angles[k]
            beta  = self.el_angles[(k + 1) % self.n_el]
            grads.append(self._grad_phi(alpha, beta))

        rows: list[np.ndarray] = []
        self._meas_idx: list[tuple[int, int]] = []

        for k in range(self.n_el):
            gx_d, gy_d = grads[k]
            for m in range(self.n_el):
                # Exclude self-measurements (shared electrodes)
                if m == k or m == (k + 1) % self.n_el:
                    continue
                gx_m, gy_m = grads[m]
                rows.append(-(gx_d * gx_m + gy_d * gy_m) * self.pixel_area)
                self._meas_idx.append((k, m))

        self.J = np.array(rows, dtype=np.float32)   # (224, 1024)
        assert self.J.shape[0] == 224, (
            f"Protocol mismatch: expected 224 rows, got {self.J.shape[0]}"
        )

        # The Fourier-series solution is valid only inside the disc (r < R).
        # Zero out columns for outside-tank pixels so J is physically consistent.
        # (Those columns are always premultiplied by Δσ=0, so results are unchanged.)
        self.J[:, ~self.tank_mask.ravel()] = 0.0

    # ── Batch data generation ─────────────────────────────────────────────────

    def generate_batch(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate n_samples random anomaly instances.

        Each anomaly is a disc fully contained within the 50 mm tank.
        The Born-approximation forward pass is a single batched matmul.

        Returns
        -------
        delta_v : float32 array, shape (n_samples, 224)
        masks   : float32 array, shape (n_samples, 1, img_sz, img_sz)
        """
        n = self.img_sz
        M = n * n  # 1024

        # ── Random anomaly geometry ──────────────────────────────────────────
        r_anom   = rng.uniform(self.r_min, self.r_max, size=n_samples)
        max_c    = np.maximum(self.R - r_anom, 1e-3)   # keep anomaly fully inside
        phi_c    = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
        # Uniform sampling inside disc: r = sqrt(u) * max_c
        r_c      = np.sqrt(rng.uniform(0.0, 1.0, size=n_samples)) * max_c
        x0 = r_c * np.cos(phi_c)   # (N,)
        y0 = r_c * np.sin(phi_c)   # (N,)

        # ── Binary conductivity-contrast masks ───────────────────────────────
        xx_f = self.xx.ravel()[None, :]          # (1, M)
        yy_f = self.yy.ravel()[None, :]          # (1, M)
        dist2 = (xx_f - x0[:, None])**2 + (yy_f - y0[:, None])**2  # (N, M)
        inside_anom = dist2 <= (r_anom[:, None]**2)                 # (N, M) bool
        inside_tank = self.tank_mask.ravel()[None, :]               # (1, M) bool
        sigma = (inside_anom & inside_tank).astype(np.float32)      # (N, M)

        # ── Forward pass: ΔV = Δσ @ J^T  ────────────────────────────────────
        delta_v = sigma @ self.J.T   # (N, 224)

        # ── Proportional Gaussian noise ──────────────────────────────────────
        if self.noise_std > 0.0:
            sig_rms = np.std(delta_v, axis=1, keepdims=True) + 1e-10
            delta_v += rng.normal(
                0.0, self.noise_std * sig_rms, size=delta_v.shape
            ).astype(np.float32)

        masks = sigma.reshape(n_samples, 1, n, n)
        return delta_v.astype(np.float32), masks.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class EITDataset(Dataset):
    """
    Pre-generates and caches all (ΔV, mask) pairs for a given split.

    Memory footprint for 10 000 samples:
        delta_v : 10 000 × 224 × 4 B ≈  9 MB
        masks   : 10 000 × 1024 × 4 B ≈ 40 MB

    Normalisation is performed per-feature (each of the 224 measurements
    independently) using the training-split statistics.
    """

    def __init__(
        self,
        forward_model: EITForwardModel,
        n_samples: int,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.delta_v, self.masks = forward_model.generate_batch(n_samples, rng)
        # delta_v: (N, 224)  masks: (N, 1, 32, 32)

        self.mean_: np.ndarray | None = None
        self.std_:  np.ndarray | None = None

    # ── Normalisation ─────────────────────────────────────────────────────────

    def fit_normalizer(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-feature z-score statistics from this dataset (training split)."""
        self.mean_ = self.delta_v.mean(axis=0)         # (224,)
        self.std_  = self.delta_v.std(axis=0) + 1e-8   # (224,)
        self.delta_v = (self.delta_v - self.mean_) / self.std_
        return self.mean_, self.std_

    def apply_normalizer(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Apply externally computed (training-split) normaliser."""
        self.mean_ = mean
        self.std_  = std
        self.delta_v = (self.delta_v - mean) / std

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.delta_v)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.delta_v[idx])   # float32, (224,)
        y = torch.from_numpy(self.masks[idx])      # float32, (1, 32, 32)
        return x, y
