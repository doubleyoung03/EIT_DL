"""
EIT 64x64 – Test and Three-Color Visualisation.

For randomly selected test samples this script:
  1. Reports Final Test MSE and RMSE over the entire test set.
  2. Uses three discrete colours only:
       +1.0  -> Red   (conductive anomaly)
       -1.0  -> Blue  (resistive anomaly)
        0.0  -> White (background)
  3. Applies Gaussian smoothing + high-resolution rendering for cleaner edges.
  4. Hides values outside the 50 mm tank and overlays a black solid tank boundary.
  5. Computes localisation error (mm) using predicted anomaly centroid vs GT centre.
  5. Saves to  results/final_reconstruction.png.

Checkpoint selection
────────────────────
  --model PATH   load a specific checkpoint file.
  (no flag)      automatically load the newest  *.pth  file in checkpoint_dir/
                 (ranked by file modification time; most recent = newest run).

Usage (from project root  D:\\Yang\\EIT\\Network):
    python src/test.py
    python src/test.py --model checkpoints/best_model_64x64_20260101_0900.pth
    python src/test.py --config config.yaml --n-samples 5 --seed 0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.ndimage import gaussian_filter, zoom

sys.path.insert(0, str(Path(__file__).parent))

from dataset import build_pixel_grid, get_dataloaders
from model import EITReconstructor

_GAUSS_SIGMA = 1.0
_CLS_THRESH = 0.1
_UPSAMPLE = 4


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ──────────────────────────────────────────────────────────────────────────────

def _find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """
    Return the most recently modified  *.pth  file in checkpoint_dir.

    Raises FileNotFoundError if the directory is empty or contains no .pth files.
    This lets the user run  python src/test.py  immediately after training
    without needing to copy or remember the timestamped filename.
    """
    candidates = sorted(
        checkpoint_dir.glob("*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,   # newest first
    )
    if not candidates:
        raise FileNotFoundError(
            f"No *.pth files found in {checkpoint_dir}.\n"
            "Run src/train.py first, or pass --model <path> explicitly."
        )
    return candidates[0]


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tank_outline(ax: plt.Axes, R: float) -> None:
    """Draw the physical tank boundary as a black solid circle."""
    th = np.linspace(0.0, 2.0 * np.pi, 300)
    ax.plot(R * np.cos(th), R * np.sin(th),
            color="black", linewidth=1.4, linestyle="-", zorder=6)


def _render(
    ax: plt.Axes,
    Z: np.ndarray,
    tank_mask: np.ndarray,
    title: str,
    R: float,
) -> None:
    """
    Draw a three-class map with outside-tank region hidden.
    """
    cmap = mcolors.ListedColormap(["#2166AC", "#FFFFFF", "#B2182B"])
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    norm = mcolors.BoundaryNorm(boundaries=[-1.5, -0.5, 0.5, 1.5], ncolors=3)
    Z_masked = np.ma.array(Z, mask=~tank_mask)
    ax.imshow(
        Z_masked,
        cmap=cmap,
        norm=norm,
        origin="upper",
        extent=[-R, R, -R, R],
        interpolation="nearest",
    )
    _tank_outline(ax, R)

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("x  [mm]", fontsize=8)
    ax.set_ylabel("y  [mm]", fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def _to_three_class(z: np.ndarray, thr: float) -> np.ndarray:
    """Convert continuous map to {-1, 0, +1} by symmetric thresholding."""
    out = np.zeros_like(z, dtype=np.int8)
    out[z >= thr] = 1
    out[z <= -thr] = -1
    return out


def _centroid_mm(
    pred_hi: np.ndarray,
    polarity: float,
    thr: float,
    xx_hi: np.ndarray,
    yy_hi: np.ndarray,
    tank_mask_hi: np.ndarray,
) -> tuple[float, float]:
    """
    Compute centroid in mm from high-resolution prediction map.
    Falls back to extremum point if thresholded region is empty.
    """
    if polarity > 0:
        region = (pred_hi >= thr) & tank_mask_hi
    else:
        region = (pred_hi <= -thr) & tank_mask_hi

    if np.any(region):
        return float(xx_hi[region].mean()), float(yy_hi[region].mean())

    pred_masked = np.where(tank_mask_hi, pred_hi, np.nan)
    if polarity > 0:
        ridx = int(np.nanargmax(pred_masked))
    else:
        ridx = int(np.nanargmin(pred_masked))
    rr, cc = np.unravel_index(ridx, pred_hi.shape)
    return float(xx_hi[rr, cc]), float(yy_hi[rr, cc])


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EIT 64x64 test & visualisation")
    p.add_argument("--config",    default="config.yaml",
                   help="Path to config.yaml (default: config.yaml)")
    p.add_argument("--model",     default=None,
                   help="Path to a specific .pth checkpoint.  "
                        "If omitted, the newest file in checkpoint_dir/ is used.")
    p.add_argument("--n-samples", type=int, default=5,
                   help="Number of test samples to visualise (default: 5)")
    p.add_argument("--seed",      type=int, default=None,
                   help="Random seed for sample selection. "
                        "If omitted, samples are random each run.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R      = float(cfg["tank_radius"])
    img_sz = int(cfg["image_size"])

    # ── Resolve checkpoint path ────────────────────────────────────────────────
    if args.model is not None:
        ckpt_path = Path(args.model)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        ckpt_path = _find_latest_checkpoint(Path(cfg["checkpoint_dir"]))
        print(f"[test] Auto-selected checkpoint: {ckpt_path.name}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EITReconstructor(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[test] Loaded  epoch={ckpt.get('epoch', '?')}  "
          f"val_MSE={ckpt.get('val_mse', float('nan')):.6f}  "
          f"ts={ckpt.get('timestamp', 'unknown')}")

    # ── Reload test split (deterministic – identical seeds to train.py) ────────
    _, _, test_loader, test_ds = get_dataloaders(cfg)

    # ── Full test-set MSE / RMSE ───────────────────────────────────────────────
    criterion = nn.MSELoss()
    total_mse, n_seen = 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y  = x.to(device), y.to(device)
            loss   = criterion(model(x), y)
            total_mse += loss.item() * len(x)
            n_seen    += len(x)

    test_mse  = total_mse / n_seen
    test_rmse = math.sqrt(test_mse)
    print(f"[test] Final Test MSE  = {test_mse:.6f}")
    print(f"[test] Final Test RMSE = {test_rmse:.6f}")

    # ── Sample selection ───────────────────────────────────────────────────────
    n    = min(args.n_samples, len(test_ds))
    rng  = np.random.default_rng(args.seed)
    if args.seed is None:
        print("[test] Sample selection: random (no fixed seed)")
    else:
        print(f"[test] Sample selection seed: {args.seed}")
    idxs = rng.choice(len(test_ds), size=n, replace=False)

    # ── High-resolution grid for smoother display/localisation ────────────────
    hi_sz = img_sz * _UPSAMPLE
    xx_hi, yy_hi, tank_mask_hi = build_pixel_grid(hi_sz, R)

    # ── Batch inference ────────────────────────────────────────────────────────
    x_batch = torch.from_numpy(test_ds.delta_v[idxs]).to(device)   # (n, 224)
    with torch.no_grad():
        preds = model(x_batch).squeeze(1).cpu().numpy()             # (n, 64, 64)

    gts     = test_ds.masks[idxs, 0]                                # (n, 64, 64)
    per_mse = ((preds - gts) ** 2).mean(axis=(1, 2))                # (n,)
    loc_errs_mm = []

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n, 2, figsize=(9.5, n * 4.2))
    if n == 1:
        axes = axes[None, :]

    for row, (idx, mse_i) in enumerate(zip(idxs, per_mse)):
        pol_sign = float(test_ds.polarity[idx])
        polarity = "Conductive" if pol_sign > 0 else "Resistive"
        x0_mm = float(test_ds.x0[idx])
        y0_mm = float(test_ds.y0[idx])
        r_mm = float(test_ds.r_anom[idx])

        # Build an analytic high-resolution GT disk for smoother circle boundary.
        gt_hi = np.zeros((hi_sz, hi_sz), dtype=np.int8)
        in_anomaly = ((xx_hi - x0_mm) ** 2 + (yy_hi - y0_mm) ** 2 <= r_mm ** 2) & tank_mask_hi
        gt_hi[in_anomaly] = 1 if pol_sign > 0 else -1

        # Smooth prediction, upsample, then quantise to three classes.
        pred_s = gaussian_filter(preds[row], sigma=_GAUSS_SIGMA)
        pred_hi = zoom(pred_s, zoom=_UPSAMPLE, order=1)
        pred_cls_hi = _to_three_class(pred_hi, thr=_CLS_THRESH)

        x_pred_mm, y_pred_mm = _centroid_mm(
            pred_hi=pred_hi,
            polarity=pol_sign,
            thr=_CLS_THRESH,
            xx_hi=xx_hi,
            yy_hi=yy_hi,
            tank_mask_hi=tank_mask_hi,
        )
        loc_err_mm = math.hypot(x_pred_mm - x0_mm, y_pred_mm - y0_mm)
        loc_errs_mm.append(loc_err_mm)

        _render(
            axes[row, 0], gt_hi, tank_mask_hi,
            title=f"[{idx:04d}]  Ground Truth  ({polarity})",
            R=R,
        )
        _render(
            axes[row, 1], pred_cls_hi, tank_mask_hi,
            title=(f"[{idx:04d}]  Prediction  "
                   f"(RMSE = {math.sqrt(float(mse_i)):.4f}, "
                   f"LocErr = {loc_err_mm:.2f} mm)"),
            R=R,
        )
        print(
            f"[test] Sample {idx:04d} | GT=({x0_mm:.2f}, {y0_mm:.2f}) mm  "
            f"Pred=({x_pred_mm:.2f}, {y_pred_mm:.2f}) mm  "
            f"LocErr={loc_err_mm:.2f} mm"
        )

    loc_errs_mm = np.asarray(loc_errs_mm, dtype=np.float64)
    print(f"[test] Localisation Error Mean   = {loc_errs_mm.mean():.2f} mm")
    print(f"[test] Localisation Error Median = {np.median(loc_errs_mm):.2f} mm")
    print(f"[test] Localisation Error P90    = {np.percentile(loc_errs_mm, 90):.2f} mm")

    fig.suptitle(
        f"EIT 64x64 Reconstruction  |  Test MSE = {test_mse:.5f}  "
        f"RMSE = {test_rmse:.5f}\n"
        "Red = Conductive (+1)   |   Blue = Resistive (-1)   |   White = Background (0)\n"
        "Black solid circle = 50 mm tank boundary; outside-tank region is hidden",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()

    out_dir  = Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "final_reconstruction.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[test] Report saved -> {out_path}")


if __name__ == "__main__":
    main()
