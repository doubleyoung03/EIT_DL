"""
EIT Dataset — Adjacent-Differential Representation (208 channels)
==================================================================

Reads the 208-column adjacent-pair differential CSV produced by
``D:\\Yang\\EIT\\COMSOL4EIT\\convert_singleend_to_adjacent.py`` and feeds
normalised time-difference inputs to the reconstruction network.

Expected CSV schema
-------------------
Metadata columns (in any order, prefix-free):
    sample_id, x0_mm, y0_mm, r_mm, sigma_touch, sample_valid

Voltage columns (exactly 208, prefix ``dV_src``):
    dV_src01_snk02_p03p04 … dV_src16_snk01_p14p15

    Each column is a single adjacent-pair differential voltage in volts:

        dV[inj, k] = V[inj, (k+1) mod 16] − V[inj, k]

    with injections that touch either electrode of the pair excluded
    (16 injections × 13 valid pairs = 208 measurements).

Reference voltage CSV
---------------------
One row, same 208 ``dV_src*_p*p*`` columns, measured/simulated on a
homogeneous background without any anomaly.  Used for the standard
normalised time-difference input:

    dv_norm = (dV_measured − dV_ref) / |dV_ref|

See ``adjacent_diff.py`` for the physics / mask derivation.

Neutral-channel floor
---------------------
Adjacent-diff magnitudes span several decades across the tank cross
section.  Channels whose reference |dV_ref| is below an adaptive floor
(default p1 of the reference distribution, with an absolute minimum of
1 nV) are treated as "neutral" and forced to zero in dv_norm to avoid
dividing by near-zero values.  Both the floor and the resulting neutral
mask are stored on the dataset and on the checkpoint so that the
inference-time preprocessing hits the exact same set of channels.

Splitting (sklearn.model_selection.train_test_split)
----------------------------------------------------
get_dataloaders() does:
  Step 1 — 80 / 20  ->  train_val_idx  /  test_idx     (random_state=42)
  Step 2 — 90 / 10  ->  train_idx      /  val_idx      (within train_val)
  Totals: ~72% train | ~8% val | 20% test
Dataset size is determined dynamically from len(dataframe) — never hardcoded.
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

from adjacent_diff import (
    N_VALID,
    adj_column_names,
    compute_adj_ref_floor,
)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing constants
# ──────────────────────────────────────────────────────────────────────────────
# MUST stay in sync with Implementation/inference.py and the probe script in
# COMSOL4EIT/diff_imaging_probe.py.  See each repo's 2026-04-22 CHANGELOG.
_ADJ_FLOOR_PERCENTILE: float = 1.0   # |dV_ref| percentile -> neutral threshold
_ADJ_FLOOR_ABS_MIN: float    = 1e-9  # volts, hard lower bound for the floor
_DV_CLIP: float              = 700.0 # empirical safety clip on dv_norm


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader seed helper
# ──────────────────────────────────────────────────────────────────────────────

def _seed_worker(worker_id: int) -> None:
    """Keep numpy/python RNG deterministic per worker (num_workers > 0)."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Pixel-grid builder (shared with test.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_pixel_grid(
    image_size: int,
    tank_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatial coordinates + tank mask for the square pixel grid.

    Row 0 is the highest y value (standard image orientation).

    Returns
    -------
    xx, yy    : float64 arrays of shape (image_size, image_size)
    tank_mask : bool    array  of shape (image_size, image_size)
                True for pixel centres inside the tank circle.
    """
    n = image_size
    R = tank_radius
    coords = (np.arange(n) + 0.5) * (2.0 * R / n) - R
    xx, yy = np.meshgrid(coords, coords[::-1])
    return xx, yy, (xx**2 + yy**2) <= R**2


# ──────────────────────────────────────────────────────────────────────────────
# CSV column-detection helpers
# ──────────────────────────────────────────────────────────────────────────────

_EXPECTED_ADJ_COLS: list[str] = adj_column_names()


def _extract_dv_columns(df: pd.DataFrame, source_tag: str) -> list[str]:
    """Return the 208 ``dV_src*_p*p*`` column names in canonical order.

    Raises if the count is not exactly 208 or if any canonical name is
    missing from ``df``.
    """
    dv_cols = [c for c in df.columns if c.startswith("dV_src")]
    if len(dv_cols) != N_VALID:
        raise ValueError(
            f"[{source_tag}] expected {N_VALID} dV_src columns, found {len(dv_cols)}. "
            "Did you run `convert_singleend_to_adjacent.py` on the CSV?"
        )
    missing = [c for c in _EXPECTED_ADJ_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{source_tag}] CSV is missing {len(missing)} canonical dV columns; "
            f"first missing: {missing[0]!r}"
        )
    return _EXPECTED_ADJ_COLS


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class EITDataset(Dataset):
    """
    Exposes (dv_scaled, mask) tensor pairs on a dataframe slice.

    Processing pipeline
    -------------------
    1. dv_raw    = (N, 208) adjacent-diff voltages pulled from 208 dV columns
    2. dv_norm   = (dv_raw − dv_ref) / |dv_ref|, with adaptive floor
    3. neutral-channel zeroing + |dv| clipping to ±_DV_CLIP
    4. StandardScaler (fitted on the training split only)
    5. 64×64 target mask from (x0, y0, r, sigma_touch) + tank geometry:
         inside anomaly AND inside tank circle: +1 (conductive) or −1 (resistive)
         all other pixels                     :  0

    Parameters
    ----------
    df              : pandas DataFrame — a contiguous row-slice of the CSV.
    config          : dict with keys ``image_size`` and ``tank_radius``.
    dv_ref_208      : (208,) float64 array — masked homogeneous-medium reference.
    neutral_mask_208: (208,) bool   array — True for channels to zero.
    floor           : float          — adaptive |dV_ref| floor (volts).
    scaler          : pre-fitted StandardScaler, or None to fit on this split.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: dict,
        dv_ref_208: np.ndarray,
        neutral_mask_208: np.ndarray,
        floor: float,
        scaler: StandardScaler | None = None,
    ) -> None:
        img_sz = int(config["image_size"])
        R      = float(config["tank_radius"])

        xx, yy, tank_mask = build_pixel_grid(img_sz, R)

        # ── Pull dV columns in canonical order ───────────────────────────────
        dv_cols = _extract_dv_columns(df, source_tag="dataset")
        dv_raw = df[dv_cols].values.astype(np.float64)     # (N, 208)

        # ── Normalised time-difference: dv_norm = (dV − dV_ref) / |dV_ref| ───
        ref_abs = np.abs(dv_ref_208)
        zero_cols = neutral_mask_208 | (ref_abs < floor)
        ref_abs_safe = np.where(zero_cols, 1.0, ref_abs)
        dv_norm = (dv_raw - dv_ref_208[None, :]) / ref_abs_safe[None, :]
        dv_norm[:, zero_cols] = 0.0
        np.clip(dv_norm, -_DV_CLIP, _DV_CLIP, out=dv_norm)

        # ── StandardScaler (fit on first construction, reuse thereafter) ─────
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(dv_norm)
        self.scaler = scaler
        dv_scaled = scaler.transform(dv_norm).astype(np.float32)

        # ── Target masks (image_size × image_size) ───────────────────────────
        x0_a  = df["x0_mm"].values
        y0_a  = df["y0_mm"].values
        r_a   = df["r_mm"].values
        sig_a = df["sigma_touch"].values

        tgt = np.where(np.abs(sig_a - 50.0) < 1.0, 1.0, -1.0).astype(np.float32)

        N = len(df)
        xx_f  = xx.ravel()[None, :]
        yy_f  = yy.ravel()[None, :]
        dist2 = (xx_f - x0_a[:, None])**2 + (yy_f - y0_a[:, None])**2
        in_an = (dist2 <= r_a[:, None]**2) & tank_mask.ravel()[None, :]
        pv    = (tgt[:, None] * in_an).astype(np.float32)

        self.delta_v = dv_scaled                                 # (N, 208)
        self.masks   = pv.reshape(N, 1, img_sz, img_sz)          # (N, 1, H, W)

        self.x0       = x0_a
        self.y0       = y0_a
        self.r_anom   = r_a
        self.polarity = tgt                                       # +1 or -1

    def __len__(self) -> int:
        return len(self.delta_v)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.delta_v[idx]),   # float32 (208,)
            torch.from_numpy(self.masks[idx]),     # float32 (1, H, W)
        )


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    config: dict,
    csv_path: Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, EITDataset]:
    """Build train / val / test DataLoaders from the adjacent-diff EIT CSV.

    The CSV must already exist at ``data_dir / csv_filename`` (or at
    ``csv_path``).  A FileNotFoundError is raised immediately if the file
    is missing.  Dataset size is determined dynamically from the CSV.

    Splitting
    ---------
    Step 1: 80 / 20  ->  train_val_idx  /  test_idx
    Step 2: 90 / 10  ->  train_idx      /  val_idx

    StandardScaler is fitted exclusively on the training split.

    Returns
    -------
    train_loader, val_loader, test_loader, test_dataset
        ``test_dataset`` is returned so test.py can access per-sample
        metadata (x0, y0, r, polarity) and the fitted scaler.  The floor
        and neutral mask are exposed as attributes on the dataset object
        for train.py to pack into the checkpoint.
    """
    if csv_path is None:
        csv_path = Path(config["data_dir"]) / config["csv_filename"]

    if not csv_path.exists():
        raise FileNotFoundError(
            f"EIT adjacency-diff CSV not found: {csv_path}\n"
            "Produce it via "
            "`python D:\\Yang\\EIT\\COMSOL4EIT\\convert_singleend_to_adjacent.py "
            "--dataset <original_single_ended.csv>`."
        )

    # ── Load reference voltages (homogeneous-medium baseline, 208 ch) ─────────
    ref_csv = config.get("reference_voltage_csv")
    if ref_csv is None:
        raise ValueError(
            "config must contain 'reference_voltage_csv' pointing to the "
            "homogeneous-medium adjacent-diff reference CSV."
        )
    ref_path = Path(config["data_dir"]) / ref_csv
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference voltage CSV not found: {ref_path}\n"
            "Produce it via `convert_singleend_to_adjacent.py` on the "
            "single-ended reference file."
        )

    ref_df = pd.read_csv(ref_path)
    ref_cols = _extract_dv_columns(ref_df, source_tag="reference")
    dv_ref_208 = ref_df[ref_cols].values.astype(np.float64).ravel()
    floor = compute_adj_ref_floor(
        dv_ref_208,
        percentile=_ADJ_FLOOR_PERCENTILE,
        min_floor=_ADJ_FLOOR_ABS_MIN,
    )
    neutral_mask_208 = np.abs(dv_ref_208) < floor
    print(
        f"[dataset] Reference loaded: {ref_path}  "
        f"(|dV_ref| median={np.median(np.abs(dv_ref_208)):.3e} V, "
        f"p{_ADJ_FLOOR_PERCENTILE:g} floor={floor:.3e} V, "
        f"neutral={int(neutral_mask_208.sum())}/{N_VALID})"
    )

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"[dataset] Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    n_total = len(df)
    print(f"[dataset] {n_total} samples detected, {len(df.columns)} columns.")

    all_idx = np.arange(n_total)

    # ── Step 1: 80 / 20  train_val / test ────────────────────────────────────
    tv_idx, te_idx = train_test_split(
        all_idx, test_size=0.20, random_state=42, shuffle=True
    )

    # ── Step 2: 90 / 10  train / val (within the 80%) ────────────────────────
    tr_idx, va_idx = train_test_split(
        tv_idx, test_size=0.10, random_state=42, shuffle=True
    )

    s_tr, s_va, s_te = set(tr_idx), set(va_idx), set(te_idx)
    assert s_tr & s_te == set(), "DATA LEAK: train / test index overlap!"
    assert s_va & s_te == set(), "DATA LEAK: val  / test index overlap!"
    assert s_tr & s_va == set(), "DATA LEAK: train / val  index overlap!"

    print(f"[dataset] Split -> train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

    # ── Build datasets (scaler fitted on train split only) ────────────────────
    train_ds = EITDataset(
        df.iloc[tr_idx].reset_index(drop=True), config,
        dv_ref_208, neutral_mask_208, floor, scaler=None,
    )
    val_ds = EITDataset(
        df.iloc[va_idx].reset_index(drop=True), config,
        dv_ref_208, neutral_mask_208, floor, scaler=train_ds.scaler,
    )
    test_ds = EITDataset(
        df.iloc[te_idx].reset_index(drop=True), config,
        dv_ref_208, neutral_mask_208, floor, scaler=train_ds.scaler,
    )

    # Expose preprocessing metadata on the returned dataset (train.py packs
    # these into the checkpoint so inference can reproduce byte-for-byte).
    for ds in (train_ds, val_ds, test_ds):
        ds.adj_floor = float(floor)
        ds.adj_neutral_mask = neutral_mask_208.astype(np.bool_)
        ds.dv_ref = dv_ref_208.astype(np.float64)

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
