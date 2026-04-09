"""
EIT 64x64 – Test & Smooth Polarised Visualisation.

For 5 randomly selected test samples this script:
  1. Reports Final Test MSE and RMSE over the entire test set.
  2. Applies scipy.ndimage.gaussian_filter (sigma=1.0) to each 64x64 prediction.
  3. Renders GT vs. smoothed Prediction using plt.contourf (50 levels, 'RdBu_r'):
       +1.0  -> Deep Red    (conductive anomaly)
       -1.0  -> Deep Blue   (resistive  anomaly)
        0.0  -> White       (background / outside-tank)
  4. Overlays a white dashed circle at the 50 mm tank boundary on every subplot.
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
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))

from dataset import build_pixel_grid, get_dataloaders
from model import EITReconstructor

_GAUSS_SIGMA = 1.0   # smoothing radius applied to 64x64 predictions


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
    """White dashed circle at the 50 mm physical tank boundary."""
    th = np.linspace(0.0, 2.0 * np.pi, 300)
    ax.plot(R * np.cos(th), R * np.sin(th),
            color="white", linewidth=1.8, linestyle="--", zorder=5)


def _render(
    ax: plt.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str,
    R: float,
) -> plt.cm.ScalarMappable:
    """
    Draw a polarised [-1, 1] field using contourf (50 filled bands, RdBu_r).

    Colour convention
    ─────────────────
    TwoSlopeNorm with vcenter=0 ensures:
        +1.0  Deep Red    (conductive anomaly)
        -1.0  Deep Blue   (resistive  anomaly)
         0.0  White       (background)
    """
    norm   = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    levels = np.linspace(-1.0, 1.0, 51)      # 50 filled colour bands

    cf = ax.contourf(X, Y, Z, levels=levels, cmap="RdBu_r", norm=norm)
    _tank_outline(ax, R)

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("x  [mm]", fontsize=8)
    ax.set_ylabel("y  [mm]", fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    return cf


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
    p.add_argument("--seed",      type=int, default=0,
                   help="Random seed for sample selection (default: 0)")
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
    idxs = rng.choice(len(test_ds), size=n, replace=False)

    # ── Coordinate grid for contourf (matches the 64x64 pixel layout) ─────────
    coords = (np.arange(img_sz) + 0.5) * (2.0 * R / img_sz) - R
    X, Y   = np.meshgrid(coords, coords[::-1])   # (64, 64) each

    # ── Batch inference ────────────────────────────────────────────────────────
    x_batch = torch.from_numpy(test_ds.delta_v[idxs]).to(device)   # (n, 224)
    with torch.no_grad():
        preds = model(x_batch).squeeze(1).cpu().numpy()             # (n, 64, 64)

    gts     = test_ds.masks[idxs, 0]                                # (n, 64, 64)
    per_mse = ((preds - gts) ** 2).mean(axis=(1, 2))                # (n,)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n, 2, figsize=(9.5, n * 4.2))
    if n == 1:
        axes = axes[None, :]

    for row, (idx, mse_i) in enumerate(zip(idxs, per_mse)):
        polarity = "Conductive" if test_ds.polarity[idx] > 0 else "Resistive"
        gt       = gts[row]
        pred_s   = gaussian_filter(preds[row], sigma=_GAUSS_SIGMA)

        cf_gt = _render(
            axes[row, 0], X, Y, gt,
            title=f"[{idx:04d}]  Ground Truth  ({polarity})",
            R=R,
        )
        plt.colorbar(cf_gt, ax=axes[row, 0], fraction=0.046, pad=0.04)

        cf_pr = _render(
            axes[row, 1], X, Y, pred_s,
            title=(f"[{idx:04d}]  Prediction  "
                   f"(RMSE = {math.sqrt(float(mse_i)):.4f})"
                   f"  [gauss sigma={_GAUSS_SIGMA}]"),
            R=R,
        )
        plt.colorbar(cf_pr, ax=axes[row, 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"EIT 64x64 Reconstruction  |  Test MSE = {test_mse:.5f}  "
        f"RMSE = {test_rmse:.5f}\n"
        "Deep Red = Conductive (+1.0)   |   Deep Blue = Resistive (-1.0)"
        "   |   White = Background (0.0)\n"
        "White dashed circle = 50 mm tank boundary",
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
