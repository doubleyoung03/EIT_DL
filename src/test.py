"""
Evaluation & visualisation for the EIT reconstruction pipeline.

For 5 randomly selected test samples this script creates a side-by-side
comparison plot with two subplots per row:

    Subplot 1 – Ground Truth  : 32×32 binary mask (viridis)
    Subplot 2 – Prediction    : sigmoid(logits) ≥ 0.5 thresholded mask (viridis)

Both subplots carry a white circle overlay that marks the physical 50 mm
tank boundary, and individual colour-bars for visual clarity.

Output: results/comparison_report.png

Usage (from project root):
    python src/test.py
    python src/test.py --config config.yaml --n-samples 5 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend – safe on all platforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from model import EITReconstructor


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tank_circle(ax: plt.Axes, R_mm: float = 50.0) -> None:
    """
    Overlay a white circle outline at the physical tank boundary.

    Because the image is displayed with extent=[−R, R, −R, R], the tank
    boundary is simply a circle centred at (0, 0) with radius R in the
    plot's coordinate system [mm].
    """
    circle = mpatches.Circle(
        (0.0, 0.0), R_mm,
        fill=False, edgecolor="white", linewidth=1.8, linestyle="--",
    )
    ax.add_patch(circle)


def _show_image(
    ax: plt.Axes,
    image: np.ndarray,
    title: str,
    R_mm: float,
    cmap: str = "viridis",
) -> None:
    """Render a 32×32 image with physical [mm] axes, tank boundary, and colour-bar."""
    extent = [-R_mm, R_mm, -R_mm, R_mm]          # [left, right, bottom, top]
    im = ax.imshow(
        image, cmap=cmap, vmin=0.0, vmax=1.0,
        extent=extent, origin="upper", aspect="equal",
    )
    _tank_circle(ax, R_mm)
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("x  [mm]", fontsize=8)
    ax.set_ylabel("y  [mm]", fontsize=8)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EIT test visualisation")
    p.add_argument("--config",    default="config.yaml")
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--seed",      type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R_mm   = float(cfg["tank_radius_mm"])

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = Path(cfg["checkpoint_dir"]) / cfg["best_model_name"]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Run src/train.py first."
        )

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EITReconstructor(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[test] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
          f"(val_loss = {ckpt.get('val_loss', float('nan')):.5f})")

    # ── Load test data ────────────────────────────────────────────────────────
    data_path = Path(cfg["data_dir"]) / "test_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {data_path}. "
            "Run src/train.py first."
        )
    data    = np.load(data_path)
    delta_v = data["delta_v"]   # (N_test, 224) – already normalised
    masks   = data["masks"]     # (N_test, 1, 32, 32)

    # ── Sample selection ──────────────────────────────────────────────────────
    n = min(args.n_samples, len(delta_v))
    rng  = np.random.default_rng(args.seed)
    idxs = rng.choice(len(delta_v), size=n, replace=False)

    # ── Compute predictions ───────────────────────────────────────────────────
    x_batch = torch.from_numpy(delta_v[idxs]).to(device)   # (n, 224)
    with torch.no_grad():
        logits = model(x_batch)                             # (n, 1, 32, 32)
    probs       = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (n, 32, 32)
    preds_bin   = (probs >= 0.5).astype(np.float32)               # (n, 32, 32)
    gts         = masks[idxs, 0]                                   # (n, 32, 32)

    # ── Compute pixel-wise accuracy per sample ────────────────────────────────
    accuracies = (preds_bin == gts).mean(axis=(1, 2)) * 100.0   # (n,) %

    # ── Build figure ──────────────────────────────────────────────────────────
    cmap = "viridis"
    fig, axes = plt.subplots(n, 2, figsize=(7.5, n * 3.4))
    if n == 1:
        axes = axes[None, :]   # ensure 2-D indexing

    for row, (idx, acc) in enumerate(zip(idxs, accuracies)):
        _show_image(
            axes[row, 0],
            gts[row],
            title=f"Sample {idx:04d}  –  Ground Truth",
            R_mm=R_mm,
            cmap=cmap,
        )
        _show_image(
            axes[row, 1],
            preds_bin[row],
            title=f"Sample {idx:04d}  –  Prediction  (acc = {acc:.1f} %)",
            R_mm=R_mm,
            cmap=cmap,
        )

    fig.suptitle(
        "EIT Reconstruction Baseline  –  Ground Truth vs MLP Prediction\n"
        "(white dashed circle = 50 mm tank boundary)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison_report.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    mean_acc = accuracies.mean()
    print(f"[test] Mean pixel accuracy over {n} samples: {mean_acc:.1f} %")
    print(f"[test] Report saved -> {out_path}")


if __name__ == "__main__":
    main()
