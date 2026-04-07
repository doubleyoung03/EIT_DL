"""
Training script for the EIT reconstruction MLP.

Usage (from the project root):
    python src/train.py
    python src/train.py --config config.yaml

What it does
------------
1. Builds the analytical EIT forward model and sensitivity matrix.
2. Pre-generates train / val / test synthetic datasets.
3. Trains with BCEWithLogitsLoss + Adam + CosineAnnealingLR.
4. Saves the best checkpoint (lowest val loss) to checkpoints/best_model.pth.
5. Writes normaliser statistics and raw test data to data/test_data.npz so
   that test.py can run without re-generating anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Allow sibling-module imports when run as a script
sys.path.insert(0, str(Path(__file__).parent))

from dataset import EITDataset, EITForwardModel
from model import EITReconstructor


# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EIT MLP baseline")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return p.parse_args()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_n    = 0
    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            loss   = criterion(model(x), y)
            total_loss += loss.item() * len(x)
            total_n    += len(x)
    return total_loss / total_n


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    # ── Directories ───────────────────────────────────────────────────────────
    for d in (cfg["data_dir"], cfg["results_dir"], cfg["checkpoint_dir"]):
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Forward model (precomputes J once) ────────────────────────────────────
    print("[train] Building EIT forward model & sensitivity matrix ...")
    fwd = EITForwardModel(cfg)
    print(f"[train] Sensitivity matrix J: {fwd.J.shape}  "
          f"(pixel_area = {fwd.pixel_area:.2f} mm^2)")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print(f"[train] Generating {cfg['n_train']} training samples ...")
    train_ds = EITDataset(fwd, cfg["n_train"], seed=42)
    norm_mean, norm_std = train_ds.fit_normalizer()
    print(f"[train] dV normaliser fitted:  "
          f"mean in [{norm_mean.min():.3e}, {norm_mean.max():.3e}]  "
          f"std in [{norm_std.min():.3e},  {norm_std.max():.3e}]")

    print(f"[train] Generating {cfg['n_val']} validation samples ...")
    val_ds = EITDataset(fwd, cfg["n_val"], seed=1337)
    val_ds.apply_normalizer(norm_mean, norm_std)

    print(f"[train] Generating {cfg['n_test']} test samples ...")
    test_ds = EITDataset(fwd, cfg["n_test"], seed=9999)
    test_ds.apply_normalizer(norm_mean, norm_std)

    # Persist test data + normaliser so test.py needs no regeneration
    np.savez(
        Path(cfg["data_dir"]) / "test_data.npz",
        delta_v   = test_ds.delta_v,    # already normalised
        masks     = test_ds.masks,
        norm_mean = norm_mean,
        norm_std  = norm_std,
        tank_mask = fwd.tank_mask.astype(np.float32),
    )
    print(f"[train] Test data saved -> {cfg['data_dir']}/test_data.npz")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EITReconstructor(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg["epochs"], eta_min=1e-6,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_cnt  = 0
    best_path     = Path(cfg["checkpoint_dir"]) / cfg["best_model_name"]

    print(f"\n{'Epoch':>6}  {'Train':>9}  {'Val':>9}  {'Best':>9}  {'LR':>10}")
    print("-" * 52)

    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_ds)

        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(
                {
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "config":       cfg,
                    "norm_mean":    norm_mean,
                    "norm_std":     norm_std,
                    "val_loss":     best_val_loss,
                },
                best_path,
            )
            tag = "*"
        else:
            patience_cnt += 1
            tag = ""

        print(
            f"{epoch:>6}  {train_loss:>9.5f}  {val_loss:>9.5f}  "
            f"{best_val_loss:>9.5f}  {current_lr:>10.2e}  {tag}"
        )

        if patience_cnt >= cfg["patience"]:
            print(f"\n[train] Early stopping triggered at epoch {epoch}.")
            break

    print(f"\n[train] Best val loss: {best_val_loss:.5f}")
    print(f"[train] Checkpoint  : {best_path}")


if __name__ == "__main__":
    main()
