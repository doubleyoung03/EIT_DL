"""
Training script – EIT 64x64 Polarised Reconstruction.

Usage (from project root  D:\\Yang\\EIT\\Network):
    python src/train.py
    python src/train.py --config config.yaml

Pipeline
────────
1.  get_dataloaders() reads the COMSOL CSV; raises FileNotFoundError if absent.
2.  Trains with nn.MSELoss().
3.  ReduceLROnPlateau  (patience=10, factor=0.5) monitors val MSE.
4.  Logs  Ep | TrMSE | TrRMSE | VaMSE | VaRMSE | LR  every epoch.
5.  Saves best checkpoint to  checkpoints/best_model_64x64_YYYYMMDD_HHMM.pth
    (timestamp is set when training begins; the same path is used throughout).
6.  Early-stops when val MSE shows no improvement for `patience` epochs.
7.  Reloads the best checkpoint and evaluates on the held-out test set.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloaders
from model import EITReconstructor


# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Seed python/numpy/torch RNG for reproducible runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Helps deterministic behavior for CUDA matmul kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EIT 64x64 reconstructor")
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimiser: torch.optim.Optimizer | None = None,
    input_noise_std: float = 0.0,
    max_grad_norm: float = 0.0,
) -> tuple[float, float]:
    """
    One forward (and optional backward) pass over a DataLoader.

    Parameters
    ----------
    optimiser : None -> evaluation mode (no gradients); not None -> training mode.

    Returns
    -------
    (mean_MSE, RMSE) aggregated over the full loader.
    """
    training = optimiser is not None
    model.train(training)

    total_loss, n_seen = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training and input_noise_std > 0:
                x = x + torch.randn_like(x) * input_noise_std
            if training:
                optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if training:
                loss.backward()
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimiser.step()
            total_loss += loss.item() * len(x)
            n_seen     += len(x)

    mse = total_loss / n_seen
    return mse, math.sqrt(mse)


# ──────────────────────────────────────────────────────────────────────────────

def save_loss_curve(
    out_path: Path,
    epochs: list[int],
    train_mse: list[float],
    val_mse: list[float],
    lrs: list[float],
) -> None:
    """Save train/val loss curve and LR trend."""
    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    ax1.plot(epochs, train_mse, color="#1f77b4", linewidth=2.0, label="Train MSE")
    ax1.plot(epochs, val_mse, color="#d62728", linewidth=2.0, label="Val MSE")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.grid(alpha=0.3, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(epochs, lrs, color="#2ca02c", linestyle="--", linewidth=1.5, label="LR")
    ax2.set_ylabel("Learning rate")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", None)
    deterministic = bool(cfg.get("deterministic", False))
    if seed is not None:
        seed_everything(int(seed), deterministic=deterministic)
        print(f"[train] seed          = {int(seed)}")
        print(f"[train] deterministic = {deterministic}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device        = {device}")
    print(f"[train] image_size    = {cfg['image_size']}  "
          f"(output neurons = {cfg['image_size'] ** 2})")

    for d in (cfg["data_dir"], cfg["results_dir"], cfg["checkpoint_dir"]):
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Timestamped run naming shared by checkpoint + plots ───────────────────
    model_name = str(cfg.get("model_name", "mlp"))
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    run_stem = f"best_model_{model_name}_{cfg['image_size']}x{cfg['image_size']}_{ts}"
    best_path = Path(cfg["checkpoint_dir"]) / f"{run_stem}.pth"
    loss_curve_path = Path(cfg["results_dir"]) / f"loss_curve_{run_stem}.png"
    print(f"[train] checkpoint    -> {best_path}")
    print(f"[train] loss curve    -> {loss_curve_path}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, test_ds = get_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model    = EITReconstructor(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] parameters    = {n_params:,}")

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr           = float(cfg["lr"]),
        weight_decay = float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode     = "min",
        patience = int(cfg["scheduler_patience"]),
        factor   = float(cfg["scheduler_factor"]),
        min_lr   = 1e-7,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_mse = float("inf")
    patience_cnt = 0
    es_patience  = int(cfg["patience"])
    min_delta = float(cfg.get("min_delta", 0.0))
    input_noise_std = float(cfg.get("input_noise_std", 0.0))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.0))

    epochs_hist: list[int] = []
    tr_hist: list[float] = []
    va_hist: list[float] = []
    lr_hist: list[float] = []

    hdr = (f"{'Ep':>5}  {'TrMSE':>10}  {'TrRMSE':>10}  "
           f"{'VaMSE':>10}  {'VaRMSE':>10}  {'LR':>9}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for epoch in range(1, int(cfg["epochs"]) + 1):
        tr_mse, tr_rmse = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimiser=optimiser,
            input_noise_std=input_noise_std,
            max_grad_norm=max_grad_norm,
        )
        va_mse, va_rmse = run_epoch(model, val_loader,   criterion, device)
        scheduler.step(va_mse)
        lr = optimiser.param_groups[0]["lr"]
        epochs_hist.append(epoch)
        tr_hist.append(tr_mse)
        va_hist.append(va_mse)
        lr_hist.append(float(lr))

        tag = ""
        if va_mse < (best_val_mse - min_delta):
            best_val_mse = va_mse
            patience_cnt = 0
            torch.save(
                {
                    "epoch":        epoch,
                    "timestamp":    ts,
                    "run_stem":     run_stem,
                    "model_name":   model_name,
                    "model_state":  model.state_dict(),
                    "config":       cfg,
                    "val_mse":      best_val_mse,
                    "loss_curve_filename": loss_curve_path.name,
                    "loss_curve_path": str(loss_curve_path),
                    "scaler_mean":  test_ds.scaler.mean_,
                    "scaler_std":   test_ds.scaler.scale_,
                },
                best_path,
            )
            tag = " *"
        else:
            patience_cnt += 1

        print(
            f"{epoch:>5}  {tr_mse:>10.6f}  {tr_rmse:>10.6f}  "
            f"{va_mse:>10.6f}  {va_rmse:>10.6f}  {lr:>9.2e}{tag}"
        )

        if patience_cnt >= es_patience:
            print(f"\n[train] Early stopping at epoch {epoch}.")
            break

    print(f"\n[train] Best val MSE  = {best_val_mse:.6f}"
          f"  (RMSE = {math.sqrt(best_val_mse):.6f})")
    print(f"[train] Checkpoint    -> {best_path}")
    save_loss_curve(
        out_path=loss_curve_path,
        epochs=epochs_hist,
        train_mse=tr_hist,
        val_mse=va_hist,
        lrs=lr_hist,
    )
    print(f"[train] Loss curve    -> {loss_curve_path}")

    # ── Final test-set evaluation with the best saved weights ─────────────────
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    te_mse, te_rmse = run_epoch(model, test_loader, criterion, device)
    print(f"[train] Test  MSE     = {te_mse:.6f}"
          f"  (RMSE = {te_rmse:.6f})")


if __name__ == "__main__":
    main()
