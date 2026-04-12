"""
Training script – EIT 64x64 Polarised Reconstruction.

Usage (from project root  D:\\Yang\\EIT\\Network):
    python src/train.py
    python src/train.py --config config.yaml

Pipeline
────────
1.  get_dataloaders() reads the COMSOL CSV; raises FileNotFoundError if absent.
2.  Trains with a configurable objective (`mse` or `composite`).
3.  ReduceLROnPlateau  (patience=10, factor=0.5) monitors val objective.
4.  Logs train/val objective + MSE/RMSE + LR every epoch.
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
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloaders
from model import EITReconstructor


# ──────────────────────────────────────────────────────────────────────────────

class CompositeReconstructionLoss(nn.Module):
    """
    Composite objective for EIT map reconstruction:
        total = w_mse * MSE + w_dice * DiceForeground + w_area * AreaRatioPenalty
    """

    def __init__(
        self,
        mse_weight: float = 0.6,
        dice_weight: float = 0.3,
        area_weight: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.dice_weight = float(dice_weight)
        self.area_weight = float(area_weight)
        self.eps = float(eps)

        if min(self.mse_weight, self.dice_weight, self.area_weight) < 0:
            raise ValueError("Composite loss weights must be non-negative.")
        if (self.mse_weight + self.dice_weight + self.area_weight) <= 0:
            raise ValueError("At least one composite loss weight must be positive.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_mse = F.mse_loss(pred, target)

        pred_fg = pred.abs().clamp(0.0, 1.0)
        gt_fg = target.abs().clamp(0.0, 1.0)
        reduce_dims = tuple(range(1, pred.ndim))

        inter = (pred_fg * gt_fg).sum(dim=reduce_dims)
        pred_area = pred_fg.sum(dim=reduce_dims)
        gt_area = gt_fg.sum(dim=reduce_dims)

        loss_dice = 1.0 - ((2.0 * inter + self.eps) / (pred_area + gt_area + self.eps))
        loss_dice = loss_dice.mean()

        area_ratio = (pred_area + self.eps) / (gt_area + self.eps)
        loss_area = (torch.log(area_ratio) ** 2).mean()

        return (
            self.mse_weight * loss_mse
            + self.dice_weight * loss_dice
            + self.area_weight * loss_area
        )


def build_loss(cfg: dict) -> tuple[nn.Module, str, dict]:
    """
    Build training objective from config.
    """
    loss_name = str(cfg.get("loss_name", "mse")).strip().lower()
    if loss_name == "mse":
        return nn.MSELoss(), "mse", {
            "mse_weight": 1.0,
            "dice_weight": 0.0,
            "area_weight": 0.0,
            "eps": 1e-6,
        }
    if loss_name == "composite":
        loss_cfg = cfg.get("loss", {}) or {}
        mse_w = float(loss_cfg.get("mse_weight", 0.6))
        dice_w = float(loss_cfg.get("dice_weight", 0.3))
        area_w = float(loss_cfg.get("area_weight", 0.1))
        eps = float(loss_cfg.get("eps", 1e-6))
        return CompositeReconstructionLoss(
            mse_weight=mse_w,
            dice_weight=dice_w,
            area_weight=area_w,
            eps=eps,
        ), "composite", {
            "mse_weight": mse_w,
            "dice_weight": dice_w,
            "area_weight": area_w,
            "eps": eps,
        }
    raise ValueError("loss_name must be either 'mse' or 'composite'.")


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
) -> tuple[float, float, float]:
    """
    One forward (and optional backward) pass over a DataLoader.

    Parameters
    ----------
    optimiser : None -> evaluation mode (no gradients); not None -> training mode.

    Returns
    -------
    (mean_loss, mean_MSE, RMSE) aggregated over the full loader.
    """
    training = optimiser is not None
    model.train(training)

    total_loss, total_mse, n_seen = 0.0, 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training and input_noise_std > 0:
                x = x + torch.randn_like(x) * input_noise_std
            if training:
                optimiser.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            mse = F.mse_loss(pred, y)
            if training:
                loss.backward()
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimiser.step()
            total_loss += loss.item() * len(x)
            total_mse  += mse.item() * len(x)
            n_seen     += len(x)

    mean_loss = total_loss / n_seen
    mean_mse = total_mse / n_seen
    return mean_loss, mean_mse, math.sqrt(mean_mse)


# ──────────────────────────────────────────────────────────────────────────────

def save_loss_curve(
    out_path: Path,
    epochs: list[int],
    train_loss: list[float],
    val_loss: list[float],
    lrs: list[float],
) -> None:
    """Save train/val objective curve and LR trend."""
    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    ax1.plot(epochs, train_loss, color="#1f77b4", linewidth=2.0, label="Train Loss")
    ax1.plot(epochs, val_loss, color="#d62728", linewidth=2.0, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Objective")
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

    criterion, loss_name, loss_cfg = build_loss(cfg)
    print(
        "[train] loss objective = "
        f"{loss_name} (mse={loss_cfg['mse_weight']:.3f}, "
        f"dice={loss_cfg['dice_weight']:.3f}, "
        f"area={loss_cfg['area_weight']:.3f})"
    )
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
    best_val_loss = float("inf")
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

    hdr = (f"{'Ep':>5}  {'TrLoss':>10}  {'TrMSE':>10}  {'TrRMSE':>10}  "
           f"{'VaLoss':>10}  {'VaMSE':>10}  {'VaRMSE':>10}  {'LR':>9}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for epoch in range(1, int(cfg["epochs"]) + 1):
        tr_loss, tr_mse, tr_rmse = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimiser=optimiser,
            input_noise_std=input_noise_std,
            max_grad_norm=max_grad_norm,
        )
        va_loss, va_mse, va_rmse = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        lr = optimiser.param_groups[0]["lr"]
        epochs_hist.append(epoch)
        tr_hist.append(tr_loss)
        va_hist.append(va_loss)
        lr_hist.append(float(lr))

        tag = ""
        if va_loss < (best_val_loss - min_delta):
            best_val_loss = va_loss
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
                    "loss_name":    loss_name,
                    "loss":         loss_cfg,
                    "val_loss":     best_val_loss,
                    "val_mse":      best_val_mse,
                    "loss_curve_filename": loss_curve_path.name,
                    "loss_curve_path": str(loss_curve_path),
                    "scaler_mean":  test_ds.scaler.mean_,
                    "scaler_std":   test_ds.scaler.scale_,
                    "reference_voltage_csv": cfg.get("reference_voltage_csv"),
                },
                best_path,
            )
            tag = " *"
        else:
            patience_cnt += 1

        print(
            f"{epoch:>5}  {tr_loss:>10.6f}  {tr_mse:>10.6f}  {tr_rmse:>10.6f}  "
            f"{va_loss:>10.6f}  {va_mse:>10.6f}  {va_rmse:>10.6f}  {lr:>9.2e}{tag}"
        )

        if patience_cnt >= es_patience:
            print(f"\n[train] Early stopping at epoch {epoch}.")
            break

    print(f"\n[train] Best val Loss = {best_val_loss:.6f}")
    print(f"[train] Best val MSE  = {best_val_mse:.6f}"
          f"  (RMSE = {math.sqrt(best_val_mse):.6f})")
    print(f"[train] Checkpoint    -> {best_path}")
    save_loss_curve(
        out_path=loss_curve_path,
        epochs=epochs_hist,
        train_loss=tr_hist,
        val_loss=va_hist,
        lrs=lr_hist,
    )
    print(f"[train] Loss curve    -> {loss_curve_path}")

    # ── Final test-set evaluation with the best saved weights ─────────────────
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    te_loss, te_mse, te_rmse = run_epoch(model, test_loader, criterion, device)
    print(f"[train] Test  Loss    = {te_loss:.6f}")
    print(f"[train] Test  MSE     = {te_mse:.6f}"
          f"  (RMSE = {te_rmse:.6f})")


if __name__ == "__main__":
    main()
