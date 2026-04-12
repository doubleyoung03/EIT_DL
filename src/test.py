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
  5. Saves to  results/reconstruction_<checkpoint_stem>.png.
  6. Appends a JSONL test record to results/test_runs.jsonl.

Checkpoint selection
────────────────────
  --model PATH   load a specific checkpoint file (highest priority).
  test_checkpoint_path in config.yaml
                load that checkpoint when --model is not provided.
  (otherwise)    automatically load the newest  *.pth  file in checkpoint_dir/
                 (ranked by file modification time; most recent = newest run).

Usage (from project root  D:\\Yang\\EIT\\Network):
    python src/test.py
    python src/test.py --model checkpoints/best_model_64x64_20260101_0900.pth
    python src/test.py --config config.yaml --n-samples 5 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
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

_DEFAULT_GAUSS_SIGMA = 1.0
_DEFAULT_CLS_THRESH = 0.1
_DEFAULT_UPSAMPLE = 4
_DEFAULT_AUTO_THRESH = False
_DEFAULT_THR_CANDIDATES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
_DEFAULT_THR_CALIB_SAMPLES = 256


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


def _resolve_threshold_candidates(cfg: dict) -> list[float]:
    """
    Resolve candidate thresholds used by auto-threshold calibration.
    """
    raw = cfg.get("vis_threshold_candidates", _DEFAULT_THR_CANDIDATES)
    if not isinstance(raw, (list, tuple)):
        raise ValueError("vis_threshold_candidates must be a list of positive floats.")
    vals = sorted({float(v) for v in raw if float(v) > 0.0})
    if not vals:
        raise ValueError("vis_threshold_candidates must contain at least one positive value.")
    return vals


def _calibrate_class_threshold(
    preds: np.ndarray,
    gts: np.ndarray,
    candidates: list[float],
) -> tuple[float, dict[str, float]]:
    """
    Pick threshold that balances overlap quality and area calibration.
    """
    gt_fg = np.abs(gts) > 0
    gt_area = np.maximum(gt_fg.sum(axis=(1, 2)).astype(np.float64), 1.0)

    best_thr = float(candidates[0])
    best_stats: dict[str, float] = {}
    best_score = -float("inf")

    for thr in candidates:
        pred_cls = _to_three_class(preds, thr=thr)
        pred_fg = pred_cls != 0
        pred_area = pred_fg.sum(axis=(1, 2)).astype(np.float64)

        inter = np.logical_and(pred_fg, gt_fg).sum(axis=(1, 2)).astype(np.float64)
        union = np.logical_or(pred_fg, gt_fg).sum(axis=(1, 2)).astype(np.float64)
        iou = inter / np.maximum(union, 1.0)
        area_ratio = pred_area / gt_area

        mean_iou = float(iou.mean())
        mean_area_ratio = float(area_ratio.mean())
        median_area_ratio = float(np.median(area_ratio))
        area_penalty = abs(math.log(max(mean_area_ratio, 1e-8)))
        score = mean_iou - 0.20 * area_penalty

        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_stats = {
                "mean_iou": mean_iou,
                "mean_area_ratio": mean_area_ratio,
                "median_area_ratio": median_area_ratio,
                "score": float(score),
            }

    return best_thr, best_stats


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
                        "If omitted, uses config test_checkpoint_path if set; "
                        "otherwise uses newest file in checkpoint_dir/.")
    p.add_argument("--n-samples", type=int, default=None,
                   help="Number of test samples to visualise. "
                        "If omitted, uses config default_test_samples.")
    p.add_argument("--seed",      type=int, default=None,
                   help="Random seed for sample selection. "
                        "If omitted, samples are random each run.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python/numpy/torch RNG for reproducible test sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _safe_float(v: float | int | np.floating | None) -> float | None:
    if v is None:
        return None
    x = float(v)
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _resolve_log_path(cfg: dict) -> Path:
    log_name = str(cfg.get("test_log_filename", "test_runs.jsonl"))
    p = Path(log_name)
    if p.is_absolute():
        return p
    return Path(cfg["results_dir"]) / p


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True))
        f.write("\n")


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    deterministic = bool(cfg.get("deterministic", False))
    seed_for_test = args.seed if args.seed is not None else cfg.get("test_seed", cfg.get("seed", None))
    if seed_for_test is not None:
        seed_for_test = int(seed_for_test)
        seed_everything(seed_for_test, deterministic=deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R      = float(cfg["tank_radius"])
    img_sz = int(cfg["image_size"])
    gauss_sigma = float(cfg.get("vis_gaussian_sigma", _DEFAULT_GAUSS_SIGMA))
    cls_thresh = float(cfg.get("vis_class_threshold", _DEFAULT_CLS_THRESH))
    upsample = int(cfg.get("vis_upsample", _DEFAULT_UPSAMPLE))
    auto_thresh = bool(cfg.get("vis_auto_threshold", _DEFAULT_AUTO_THRESH))
    thr_candidates = _resolve_threshold_candidates(cfg)
    thr_calib_samples = int(cfg.get("vis_threshold_calib_samples", _DEFAULT_THR_CALIB_SAMPLES))

    # ── Resolve checkpoint path (CLI > config > newest in checkpoint_dir) ─────
    if args.model is not None:
        ckpt_path = Path(args.model)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"[test] Using checkpoint from --model: {ckpt_path}")
    else:
        cfg_ckpt = cfg.get("test_checkpoint_path", None)
        if isinstance(cfg_ckpt, str):
            cfg_ckpt = cfg_ckpt.strip()

        if cfg_ckpt:
            ckpt_path = Path(str(cfg_ckpt))
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint from config not found: {ckpt_path}")
            print(f"[test] Using checkpoint from config: {ckpt_path}")
        else:
            ckpt_path = _find_latest_checkpoint(Path(cfg["checkpoint_dir"]))
            print(f"[test] Auto-selected checkpoint: {ckpt_path.name}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EITReconstructor(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    run_stem = str(ckpt.get("run_stem", ckpt_path.stem))
    loss_curve_name = str(ckpt.get("loss_curve_filename", f"loss_curve_{run_stem}.png"))
    print(f"[test] Loaded  epoch={ckpt.get('epoch', '?')}  "
          f"val_MSE={ckpt.get('val_mse', float('nan')):.6f}  "
          f"ts={ckpt.get('timestamp', 'unknown')}  "
          f"model={ckpt.get('model_name', cfg.get('model_name', 'unknown'))}")

    # ── Reload test split (deterministic – identical seeds to train.py) ────────
    _, val_loader, test_loader, test_ds = get_dataloaders(cfg)
    val_ds = val_loader.dataset

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

    threshold_stats = None
    if auto_thresh:
        n_calib = min(max(int(thr_calib_samples), 1), len(val_ds))
        calib_idxs = np.arange(n_calib, dtype=np.int64)
        x_calib = torch.from_numpy(val_ds.delta_v[calib_idxs]).to(device)
        with torch.no_grad():
            preds_calib = model(x_calib).squeeze(1).cpu().numpy()
        gts_calib = val_ds.masks[calib_idxs, 0]
        if gauss_sigma > 0.0:
            preds_calib = np.stack(
                [gaussian_filter(pred, sigma=gauss_sigma) for pred in preds_calib],
                axis=0,
            )
        cls_thresh, threshold_stats = _calibrate_class_threshold(
            preds=preds_calib,
            gts=gts_calib,
            candidates=thr_candidates,
        )
        print(
            "[test] Auto-threshold selected "
            f"{cls_thresh:.3f} using {n_calib} val samples "
            f"(mean IoU={threshold_stats['mean_iou']:.4f}, "
            f"mean area ratio={threshold_stats['mean_area_ratio']:.3f})"
        )

    # ── Sample selection ───────────────────────────────────────────────────────
    requested_n = args.n_samples if args.n_samples is not None else int(cfg.get("default_test_samples", 5))
    n = min(int(requested_n), len(test_ds))
    if n <= 0:
        raise ValueError("n-samples must be >= 1.")
    rng  = np.random.default_rng(seed_for_test)
    if seed_for_test is None:
        print("[test] Sample selection: random (no fixed seed)")
    else:
        print(f"[test] Sample selection seed: {seed_for_test}")
    idxs = rng.choice(len(test_ds), size=n, replace=False)

    # ── High-resolution grid for smoother display/localisation ────────────────
    hi_sz = img_sz * upsample
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
        pred_s = gaussian_filter(preds[row], sigma=gauss_sigma)
        pred_hi = zoom(pred_s, zoom=upsample, order=1)
        pred_cls_hi = _to_three_class(pred_hi, thr=cls_thresh)

        x_pred_mm, y_pred_mm = _centroid_mm(
            pred_hi=pred_hi,
            polarity=pol_sign,
            thr=cls_thresh,
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
    out_path = out_dir / f"reconstruction_{run_stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[test] Report saved -> {out_path}")

    # ── Append one run record ─────────────────────────────────────────────────
    loc_mean = float(loc_errs_mm.mean())
    loc_median = float(np.median(loc_errs_mm))
    loc_p90 = float(np.percentile(loc_errs_mm, 90))

    cfg_snapshot = {
        "input_dim": cfg.get("input_dim"),
        "image_size": cfg.get("image_size"),
        "tank_radius": cfg.get("tank_radius"),
        "data_dir": cfg.get("data_dir"),
        "csv_filename": cfg.get("csv_filename"),
        "model_name": cfg.get("model_name"),
        "model": cfg.get("model", {}),
        "lr": cfg.get("lr"),
        "weight_decay": cfg.get("weight_decay"),
        "batch_size": cfg.get("batch_size"),
        "num_workers": cfg.get("num_workers"),
        "epochs": cfg.get("epochs"),
        "scheduler_patience": cfg.get("scheduler_patience"),
        "scheduler_factor": cfg.get("scheduler_factor"),
        "patience": cfg.get("patience"),
        "min_delta": cfg.get("min_delta"),
        "input_noise_std": cfg.get("input_noise_std"),
        "max_grad_norm": cfg.get("max_grad_norm"),
        "seed": cfg.get("seed"),
        "test_seed": cfg.get("test_seed"),
        "deterministic": cfg.get("deterministic"),
        "test_checkpoint_path": cfg.get("test_checkpoint_path"),
        "vis_gaussian_sigma": cfg.get("vis_gaussian_sigma"),
        "vis_class_threshold": cfg.get("vis_class_threshold"),
        "vis_auto_threshold": cfg.get("vis_auto_threshold"),
        "vis_threshold_candidates": cfg.get("vis_threshold_candidates"),
        "vis_threshold_calib_samples": cfg.get("vis_threshold_calib_samples"),
        "vis_selected_threshold": cls_thresh,
        "vis_upsample": cfg.get("vis_upsample"),
    }
    log_record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(Path(args.config)),
        "config": cfg_snapshot,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_mse": _safe_float(ckpt.get("val_mse")),
        "run_stem": run_stem,
        "loss_curve_file": loss_curve_name,
        "reconstruction_file": out_path.name,
        "n_samples": int(n),
        "seed": seed_for_test,
        "sample_indices": [int(i) for i in idxs.tolist()],
        "metrics": {
            "test_mse": float(test_mse),
            "test_rmse": float(test_rmse),
            "loc_err_mean_mm": loc_mean,
            "loc_err_median_mm": loc_median,
            "loc_err_p90_mm": loc_p90,
        },
        "threshold_calibration": threshold_stats,
    }
    log_path = _resolve_log_path(cfg)
    _append_jsonl(log_path, log_record)
    print(f"[test] Log appended -> {log_path}")


if __name__ == "__main__":
    main()
