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
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloaders
from model import EITReconstructor


# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EIT 64x64 MLP")
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimiser: torch.optim.Optimizer | None = None,
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
            if training:
                optimiser.zero_grad()
            loss = criterion(model(x), y)
            if training:
                loss.backward()
                optimiser.step()
            total_loss += loss.item() * len(x)
            n_seen     += len(x)

    mse = total_loss / n_seen
    return mse, math.sqrt(mse)


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device        = {device}")
    print(f"[train] image_size    = {cfg['image_size']}  "
          f"(output neurons = {cfg['image_size'] ** 2})")

    for d in (cfg["data_dir"], cfg["results_dir"], cfg["checkpoint_dir"]):
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Timestamped checkpoint path  (set once at training start) ─────────────
    ts        = datetime.now().strftime("%Y%m%d_%H%M")
    best_path = Path(cfg["checkpoint_dir"]) / f"best_model_64x64_{ts}.pth"
    print(f"[train] checkpoint    -> {best_path}")

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

    hdr = (f"{'Ep':>5}  {'TrMSE':>10}  {'TrRMSE':>10}  "
           f"{'VaMSE':>10}  {'VaRMSE':>10}  {'LR':>9}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for epoch in range(1, int(cfg["epochs"]) + 1):
        tr_mse, tr_rmse = run_epoch(model, train_loader, criterion, device, optimiser)
        va_mse, va_rmse = run_epoch(model, val_loader,   criterion, device)
        scheduler.step(va_mse)
        lr = optimiser.param_groups[0]["lr"]

        tag = ""
        if va_mse < best_val_mse:
            best_val_mse = va_mse
            patience_cnt = 0
            torch.save(
                {
                    "epoch":        epoch,
                    "timestamp":    ts,
                    "model_state":  model.state_dict(),
                    "config":       cfg,
                    "val_mse":      best_val_mse,
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

    # ── Final test-set evaluation with the best saved weights ─────────────────
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    te_mse, te_rmse = run_epoch(model, test_loader, criterion, device)
    print(f"[train] Test  MSE     = {te_mse:.6f}"
          f"  (RMSE = {te_rmse:.6f})")


if __name__ == "__main__":
    main()
