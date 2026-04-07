"""
EIT Reconstruction MLP.

Architecture (as specified):
    Input  : (B, 224)  – normalised differential voltage ΔV
    Hidden : 224 → 512 → 1024 → 2048  (each: Linear → BN → ReLU → Dropout)
    Output : 2048 → 1024, reshaped to (B, 1, 32, 32) – raw logits

No sigmoid on the output; BCEWithLogitsLoss is applied during training.
At inference, apply torch.sigmoid() and threshold at 0.5.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EITReconstructor(nn.Module):
    """
    Fully-connected MLP for EIT image reconstruction.

    Parameters
    ----------
    config : dict
        Keys consumed: input_dim (224), image_size (32).
    dropout : float
        Dropout probability for hidden layers. Default 0.2.
    """

    def __init__(self, config: dict, dropout: float = 0.2) -> None:
        super().__init__()

        inp     = int(config["input_dim"])    # 224
        img_sz  = int(config["image_size"])   # 32
        out_dim = img_sz * img_sz             # 1024

        def _block(in_f: int, out_f: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        # 224 → 512 → 1024 → 2048
        self.encoder = nn.Sequential(
            _block(inp,  512),
            _block(512,  1024),
            _block(1024, 2048),
        )

        # 2048 → 1024 (logits, no activation)
        self.head = nn.Linear(2048, out_dim)

        self.img_sz = img_sz
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-uniform init for linear layers; BatchNorm defaults are fine."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 224)
            Normalised differential voltages.

        Returns
        -------
        logits : torch.Tensor, shape (B, 1, 32, 32)
            Raw logits; apply sigmoid + threshold 0.5 for binary mask.
        """
        h      = self.encoder(x)                              # (B, 2048)
        logits = self.head(h)                                 # (B, 1024)
        return logits.reshape(-1, 1, self.img_sz, self.img_sz)  # (B, 1, 32, 32)
