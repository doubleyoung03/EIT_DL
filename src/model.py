"""
EIT Reconstruction MLP – 64x64 High-Resolution Polarised Output.

Architecture
────────────
    Input         (B, 224)    normalised delta-V

    Encoder  [3 blocks, each = Linear -> BatchNorm1d -> ReLU -> Dropout(p)]
        224  ->  512
        512  ->  1024
        1024 ->  2048

    Head     [final linear + bounded activation]
        2048 ->  4096   (= 64 * 64)
        nn.Tanh()       bounds output to [-1, 1]

    Reshape  (B, 4096) -> (B, 1, 64, 64)

Output semantics
────────────────
    +1.0  conductive anomaly pixel
    -1.0  resistive  anomaly pixel
     0.0  background / outside-tank
Trained with nn.MSELoss() against continuous targets in {-1, 0, +1}.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EITReconstructor(nn.Module):
    """
    Fully-connected MLP for 64x64 polarised EIT reconstruction.

    Parameters
    ----------
    config : dict
        Required keys:
            input_dim   : int         – 224
            image_size  : int         – 64   (output = 64*64 = 4096 neurons)
            hidden_dims : list[int]   – [512, 1024, 2048]
            dropout     : float       – 0.2
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        in_dim      = int(config["input_dim"])
        img_sz      = int(config["image_size"])
        hidden_dims = list(config["hidden_dims"])
        p_drop      = float(config.get("dropout", 0.2))
        out_dim     = img_sz * img_sz                    # 64*64 = 4096

        def _block(in_f: int, out_f: int) -> nn.Sequential:
            """Standard MLP block: Linear -> BatchNorm1d -> ReLU -> Dropout."""
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_drop),
            )

        # Encoder: build dynamically from hidden_dims list
        dims    = [in_dim] + hidden_dims              # [224, 512, 1024, 2048]
        encoder = [_block(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.encoder = nn.Sequential(*encoder)

        # Head: 2048 -> 4096, bounded to [-1, 1] by Tanh
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], out_dim),
            nn.Tanh(),
        )

        self.img_sz = img_sz
        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Kaiming-uniform init for all Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, 224)  – normalised differential voltages.

        Returns
        -------
        Tensor  (B, 1, 64, 64)  – pixel predictions in [-1, 1].
        """
        h   = self.encoder(x)                              # (B, 2048)
        out = self.head(h)                                 # (B, 4096)
        return out.reshape(-1, 1, self.img_sz, self.img_sz)  # (B, 1, 64, 64)
