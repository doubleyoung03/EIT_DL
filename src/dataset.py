"""
EIT Dataset – Polarised Anomalies, 64x64 High-Resolution Output.

Expected CSV schema  (one row = one sample, 517 columns total)
──────────────────────────────────────────────────────────────
Col  0   x0_mm        – anomaly centre x  [mm]
Col  1   y0_mm        – anomaly centre y  [mm]
Col  2   r_mm         – anomaly radius    [mm]
Col  3   sigma_bg     – background conductivity [S/m]
Col  4   sigma_touch  – anomaly conductivity [S/m]
             ~50    -> conductive -> target pixel value = +1.0
             ~1e-9  -> resistive  -> target pixel value = -1.0
Cols 5–260   V_sample_000..255  – 16x16 voltage matrix, drive-major order
Cols 261–516 V_ref_000..255     – homogeneous-reference voltage matrix

The CSV is produced externally (e.g. by COMSOL) and must exist at the path
specified by  data_dir / csv_filename  in config.yaml before training starts.
No synthetic data generation is performed here.

Self-measurement mask
─────────────────────
In the 16x16 adjacency matrix (row = drive k, col = measure m):
  invalid when  m == k  OR  m == (k+1) % 16   (shared electrode)
16 drives x 2 invalid each = 32 masked entries  ->  224 valid measurements.

Splitting  (sklearn.model_selection.train_test_split)
─────────────────────────────────────────────────────
get_dataloaders() does:
  Step 1 – 80 / 20  ->  train_val_idx  /  test_idx     (random_state=42)
  Step 2 – 90 / 10  ->  train_idx      /  val_idx      (within train_val)
  Totals: ~72% train | ~8% val | 20% test
Dataset size is determined dynamically from len(dataframe) – never hardcoded.
Three assert statements verify ZERO index overlap across all splits.
StandardScaler is fitted EXCLUSIVELY on the training indices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Self-measurement boolean mask  (computed once at module load)
# ──────────────────────────────────────────────────────────────────────────────

def _build_valid_mask(n_el: int = 16) -> np.ndarray:
    """
    Return a flat boolean array of length n_el^2.
    True  => valid  adjacent-adjacent measurement.
    False => self-measurement  (m == k  or  m == (k+1) % n_el).
    Result for n_el=16: 16 * (16-2) = 224 True entries out of 256.
    """
    mask = np.ones((n_el, n_el), dtype=bool)
    for k in range(n_el):
        mask[k, k]               = False   # drive pair == measure pair
        mask[k, (k + 1) % n_el] = False   # measure pair shares sink electrode
    return mask.ravel()


_VALID_MASK = _build_valid_mask(16)   # (256,) bool, module-level cache


# ──────────────────────────────────────────────────────────────────────────────
# Pixel-grid builder  (shared by EITDataset and test.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_pixel_grid(
    image_size: int,
    tank_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build spatial coordinate arrays for the square pixel grid.

    Pixel centres span (-tank_radius, +tank_radius) mm in both x and y.
    Row 0 = highest y value (top of image), matching standard image orientation.

    Returns
    -------
    xx, yy    : float64 arrays of shape (image_size, image_size)
    tank_mask : bool    array  of shape (image_size, image_size)
                        True for pixels whose centre lies inside the tank circle.
    """
    n      = image_size
    R      = tank_radius
    coords = (np.arange(n) + 0.5) * (2.0 * R / n) - R
    xx, yy = np.meshgrid(coords, coords[::-1])
    return xx, yy, (xx**2 + yy**2) <= R**2


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class EITDataset(Dataset):
    """
    Wraps a pandas DataFrame slice and exposes (delta_v, mask) tensor pairs.

    Processing pipeline
    ───────────────────
    1. delta_v_raw  = V_sample - V_ref                     shape (N, 256)
    2. Apply _VALID_MASK  ->  delta_v_224                  shape (N, 224)
    3. StandardScaler (fitted on training split only)      shape (N, 224)
    4. Build 64x64 target mask from geometry + sigma_touch:
         inside anomaly AND inside tank circle: +1.0 (conductive) or -1.0 (resistive)
         all other pixels                     :  0.0

    Parameters
    ----------
    df     : pandas DataFrame – a contiguous row-slice of the full CSV.
    config : dict with keys: image_size, tank_radius.
    scaler : pre-fitted StandardScaler, or None to fit on this split.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: dict,
        scaler: StandardScaler | None = None,
    ) -> None:
        img_sz = int(config["image_size"])
        R      = float(config["tank_radius"])

        xx, yy, tank_mask = build_pixel_grid(img_sz, R)

        # ── Delta-V: 256 raw  ->  224 valid measurements ─────────────────────
        vs_cols = [c for c in df.columns if c.startswith("V_sample_")]
        vr_cols = [c for c in df.columns if c.startswith("V_ref_")]
        dv_raw  = (df[vs_cols].values - df[vr_cols].values).astype(np.float32)
        dv_224  = dv_raw[:, _VALID_MASK]                   # (N, 224)

        # ── StandardScaler ────────────────────────────────────────────────────
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(dv_224)
        self.scaler = scaler
        dv_scaled   = scaler.transform(dv_224).astype(np.float32)

        # ── Target masks  (image_size x image_size) ───────────────────────────
        x0_a  = df["x0_mm"].values
        y0_a  = df["y0_mm"].values
        r_a   = df["r_mm"].values
        sig_a = df["sigma_touch"].values           # column index 4

        # Polarity: +1.0 if conductive (~50 S/m), -1.0 if resistive (~1e-9 S/m)
        tgt   = np.where(np.abs(sig_a - 50.0) < 1.0, 1.0, -1.0).astype(np.float32)

        # Vectorised mask construction over all N samples
        N     = len(df)                            # dynamic – never hardcoded
        xx_f  = xx.ravel()[None, :]                # (1, img_sz^2)
        yy_f  = yy.ravel()[None, :]
        dist2 = (xx_f - x0_a[:, None])**2 + (yy_f - y0_a[:, None])**2
        in_an = (dist2 <= r_a[:, None]**2) & tank_mask.ravel()[None, :]
        pv    = (tgt[:, None] * in_an).astype(np.float32)   # (N, img_sz^2)

        self.delta_v = dv_scaled                   # (N, 224)
        self.masks   = pv.reshape(N, 1, img_sz, img_sz)     # (N, 1, 64, 64)

        # Metadata kept for test.py visualisation
        self.x0       = x0_a
        self.y0       = y0_a
        self.r_anom   = r_a
        self.polarity = tgt                        # (N,) float32: +1 or -1

    def __len__(self) -> int:
        return len(self.delta_v)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.from_numpy(self.delta_v[idx]),   # float32 (224,)
                torch.from_numpy(self.masks[idx]))     # float32 (1, 64, 64)


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    config: dict,
    csv_path: Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, EITDataset]:
    """
    Build train / val / test DataLoaders from the COMSOL-generated EIT CSV.

    The CSV must already exist at  data_dir / csv_filename  (or at csv_path).
    A FileNotFoundError is raised immediately if the file is missing.
    Dataset size is determined dynamically from the number of rows in the CSV.

    Splitting
    ─────────
    Step 1: 80 / 20  ->  train_val_idx  /  test_idx
    Step 2: 90 / 10  ->  train_idx      /  val_idx   (within train_val)

    Three assert statements guarantee zero index overlap between all splits.
    StandardScaler is fitted exclusively on train_idx.

    Returns
    ───────
    train_loader, val_loader, test_loader, test_dataset
    test_dataset is returned so test.py can access per-sample metadata.
    """
    if csv_path is None:
        csv_path = Path(config["data_dir"]) / config["csv_filename"]

    # ── Require CSV to exist – no synthetic generation ────────────────────────
    if not csv_path.exists():
        raise FileNotFoundError(
            f"EIT CSV not found: {csv_path}\n"
            "Please place your COMSOL-exported CSV at that path before training."
        )

    # ── Load  (size determined dynamically from the file) ─────────────────────
    print(f"[dataset] Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    n_total = len(df)
    print(f"[dataset] {n_total} samples detected, {len(df.columns)} columns.")

    all_idx = np.arange(n_total)

    # ── Step 1: 80 / 20  train_val / test ────────────────────────────────────
    tv_idx, te_idx = train_test_split(
        all_idx, test_size=0.20, random_state=42, shuffle=True
    )

    # ── Step 2: 90 / 10  train / val  (within the 80%) ───────────────────────
    tr_idx, va_idx = train_test_split(
        tv_idx, test_size=0.10, random_state=42, shuffle=True
    )

    # ── Zero-overlap assertions ────────────────────────────────────────────────
    s_tr, s_va, s_te = set(tr_idx), set(va_idx), set(te_idx)
    assert s_tr & s_te == set(), "DATA LEAK: train / test index overlap!"
    assert s_va & s_te == set(), "DATA LEAK: val  / test index overlap!"
    assert s_tr & s_va == set(), "DATA LEAK: train / val  index overlap!"

    print(f"[dataset] Split -> train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

    # ── Build datasets (scaler fitted on train split only) ────────────────────
    train_ds = EITDataset(df.iloc[tr_idx].reset_index(drop=True), config, scaler=None)
    val_ds   = EITDataset(df.iloc[va_idx].reset_index(drop=True), config, scaler=train_ds.scaler)
    test_ds  = EITDataset(df.iloc[te_idx].reset_index(drop=True), config, scaler=train_ds.scaler)

    bs  = int(config["batch_size"])
    pin = torch.cuda.is_available()

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0, pin_memory=pin),
        DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin),
        DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin),
        test_ds,
    )
