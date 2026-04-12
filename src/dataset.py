"""
EIT Dataset – Polarised Anomalies, 64x64 High-Resolution Output.

Expected CSV schema  (one row = one sample, 262 columns total)
──────────────────────────────────────────────────────────────
Col  0   sample_id    – integer sample index
Col  1   x0_mm        – anomaly centre x  [mm]
Col  2   y0_mm        – anomaly centre y  [mm]
Col  3   r_mm         – anomaly radius    [mm]
Col  4   sigma_touch  – anomaly conductivity [S/m]
             ~50    -> conductive -> target pixel value = +1.0
             ~1e-9  -> resistive  -> target pixel value = -1.0
Col  5   sample_valid – validity flag (1 = valid)
Cols 6–261  V_srcXX_snkYY_chZZ – 16 drive pairs × 16 measurement channels
            (drive-major order, channels 01–16 per drive pair)

Reference voltage CSV  (one row, 256 V_src columns)
────────────────────────────────────────────────────
Homogeneous-medium baseline measured/simulated without any anomaly.
Used to compute the differential normalised input:
    dv_norm = (V_measured - V_ref) / |V_ref|

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

import random
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
# DataLoader seed helper
# ──────────────────────────────────────────────────────────────────────────────

def _seed_worker(worker_id: int) -> None:
    """
    Keep numpy/python RNG deterministic per worker.
    Useful when num_workers > 0.
    """
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


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
    Wraps a pandas DataFrame slice and exposes (v_meas, mask) tensor pairs.

    Processing pipeline
    ───────────────────
    1. v_raw    = raw voltages from V_srcXX_snkYY_chZZ columns  shape (N, 256)
    2. Apply _VALID_MASK  ->  v_224                              shape (N, 224)
    3. Differential normalization:
         dv = (v_224 - v_ref) / |v_ref|                         shape (N, 224)
    4. StandardScaler (fitted on training split only)            shape (N, 224)
    5. Build 64x64 target mask from geometry + sigma_touch:
         inside anomaly AND inside tank circle: +1.0 (conductive) or -1.0 (resistive)
         all other pixels                     :  0.0

    Parameters
    ----------
    df        : pandas DataFrame – a contiguous row-slice of the full CSV.
    config    : dict with keys: image_size, tank_radius.
    v_ref_224 : (224,) float32 array – masked homogeneous-medium reference voltages.
    scaler    : pre-fitted StandardScaler, or None to fit on this split.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: dict,
        v_ref_224: np.ndarray,
        scaler: StandardScaler | None = None,
    ) -> None:
        img_sz = int(config["image_size"])
        R      = float(config["tank_radius"])

        xx, yy, tank_mask = build_pixel_grid(img_sz, R)

        # ── Voltage measurements: 256 raw  ->  224 valid ─────────────────────
        v_cols = [c for c in df.columns if c.startswith("V_src")]
        if len(v_cols) != 256:
            raise ValueError(
                f"Expected 256 voltage columns (V_src...), found {len(v_cols)}. "
                "Check CSV column names."
            )
        v_raw  = df[v_cols].values.astype(np.float32)      # (N, 256)
        v_224  = v_raw[:, _VALID_MASK]                     # (N, 224)

        # ── Differential normalization w.r.t. homogeneous reference ───────────
        #    dv_norm = (V_measured - V_ref) / |V_ref|
        ref = v_ref_224[None, :]                           # (1, 224) broadcast
        dv_224 = (v_224 - ref) / np.abs(ref)               # (N, 224)

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

    # ── Load reference voltages (homogeneous-medium baseline) ─────────────────
    ref_csv = config.get("reference_voltage_csv")
    if ref_csv is None:
        raise ValueError(
            "config must contain 'reference_voltage_csv' pointing to the "
            "homogeneous-medium reference voltage file."
        )
    ref_path = Path(config["data_dir"]) / ref_csv
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference voltage CSV not found: {ref_path}\n"
            "Please place your homogeneous-medium voltage CSV at that path."
        )
    ref_df = pd.read_csv(ref_path)
    ref_v_cols = [c for c in ref_df.columns if c.startswith("V_src")]
    if len(ref_v_cols) != 256:
        raise ValueError(
            f"Expected 256 voltage columns in reference CSV, found {len(ref_v_cols)}."
        )
    v_ref_224 = ref_df[ref_v_cols].values.astype(np.float32).ravel()[_VALID_MASK]
    print(f"[dataset] Reference voltages loaded from {ref_path}  ({v_ref_224.shape[0]} channels)")

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
    train_ds = EITDataset(df.iloc[tr_idx].reset_index(drop=True), config, v_ref_224, scaler=None)
    val_ds   = EITDataset(df.iloc[va_idx].reset_index(drop=True), config, v_ref_224, scaler=train_ds.scaler)
    test_ds  = EITDataset(df.iloc[te_idx].reset_index(drop=True), config, v_ref_224, scaler=train_ds.scaler)

    bs  = int(config["batch_size"])
    pin = torch.cuda.is_available()
    num_workers = int(config.get("num_workers", 0))
    seed = config.get("seed", None)

    train_gen = None
    if seed is not None:
        train_gen = torch.Generator()
        train_gen.manual_seed(int(seed))

    worker_init = _seed_worker if num_workers > 0 else None

    return (
        DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            generator=train_gen,
            worker_init_fn=worker_init,
        ),
        DataLoader(
            val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            worker_init_fn=worker_init,
        ),
        DataLoader(
            test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            worker_init_fn=worker_init,
        ),
        test_ds,
    )
