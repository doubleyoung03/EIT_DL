"""
EIT Reconstruction models.

All variants map:
    input  : (B, input_dim=224)
    output : (B, 1, image_size, image_size), bounded to [-1, 1] by Tanh.

Supported model_name values (or compatible aliases):
    - "mlp", "mlp_large", "mlp_xlarge"
    - "res_mlp_large", "res_mlp_xlarge"
    - "cnn_decoder_large", "cnn_decoder_xlarge"
    - "bilstm_decoder_large", "bilstm_decoder_xlarge"
    - "transformer_decoder_large", "transformer_decoder_xlarge"
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _required_upsamples(image_size: int, latent_hw: int) -> int:
    ratio = image_size / latent_hw
    if ratio < 1 or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("image_size must be an integer multiple of latent_hw.")
    ratio_i = int(round(ratio))
    steps = int(math.log2(ratio_i)) if ratio_i > 0 else -1
    if (1 << steps) != ratio_i:
        raise ValueError("image_size/latent_hw must be a power of 2.")
    return steps


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        inner_dim = max(hidden_dim, int(hidden_dim * mlp_ratio))
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return x + h


class ConvDecoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        latent_hw: int,
        channels: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        n_up = _required_upsamples(image_size=image_size, latent_hw=latent_hw)
        if len(channels) != n_up + 1:
            raise ValueError(
                "decoder_channels length must be log2(image_size/latent_hw)+1."
            )

        layers: list[nn.Module] = []
        for i in range(n_up):
            in_c, out_c = int(channels[i]), int(channels[i + 1])
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_c,
                        out_c,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_c),
                    nn.GELU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

        layers.extend(
            [
                nn.Conv2d(channels[-1], 1, kernel_size=3, padding=1),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LegacyMLPReconstructor(nn.Module):
    def __init__(self, input_dim: int, image_size: int, model_cfg: dict) -> None:
        super().__init__()
        hidden_dims = list(model_cfg.get("hidden_dims", [512, 1024, 2048]))
        dropout = float(model_cfg.get("dropout", 0.2))
        out_dim = image_size * image_size

        layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(hidden_dims[-1], out_dim), nn.Tanh())
        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        out = self.head(h)
        return out.reshape(-1, 1, self.image_size, self.image_size)


class ResMLPReconstructor(nn.Module):
    def __init__(self, input_dim: int, image_size: int, model_cfg: dict) -> None:
        super().__init__()
        hidden_dim = int(model_cfg.get("hidden_dim", 2048))
        num_blocks = int(model_cfg.get("num_blocks", 8))
        mlp_ratio = float(model_cfg.get("mlp_ratio", 2.0))
        dropout = float(model_cfg.get("dropout", 0.25))
        out_dim = image_size * image_size

        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim), nn.Tanh())
        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        out = self.head(h)
        return out.reshape(-1, 1, self.image_size, self.image_size)


class CNNDecoderReconstructor(nn.Module):
    def __init__(self, input_dim: int, image_size: int, model_cfg: dict) -> None:
        super().__init__()
        encoder_dims = list(model_cfg.get("encoder_dims", [1024, 2048, 4096]))
        dropout = float(model_cfg.get("dropout", 0.25))
        latent_channels = int(model_cfg.get("latent_channels", 256))
        latent_hw = int(model_cfg.get("latent_hw", 8))
        decoder_channels = list(model_cfg.get("decoder_channels", [256, 128, 64, 32]))
        decoder_dropout = float(model_cfg.get("decoder_dropout", 0.10))
        if decoder_channels[0] != latent_channels:
            raise ValueError("decoder_channels[0] must equal latent_channels.")

        enc_layers: list[nn.Module] = []
        dims = [input_dim] + encoder_dims
        for i in range(len(dims) - 1):
            enc_layers.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.LayerNorm(dims[i + 1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.encoder = nn.Sequential(*enc_layers)

        latent_dim = latent_channels * latent_hw * latent_hw
        self.to_latent = nn.Linear(encoder_dims[-1], latent_dim)
        self.decoder = ConvDecoder(
            image_size=image_size,
            latent_hw=latent_hw,
            channels=decoder_channels,
            dropout=decoder_dropout,
        )
        self.latent_channels = latent_channels
        self.latent_hw = latent_hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.to_latent(h)
        z = z.reshape(-1, self.latent_channels, self.latent_hw, self.latent_hw)
        return self.decoder(z)


class BiLSTMDecoderReconstructor(nn.Module):
    def __init__(self, input_dim: int, image_size: int, model_cfg: dict) -> None:
        super().__init__()
        embed_dim = int(model_cfg.get("embed_dim", 64))
        hidden_dim = int(model_cfg.get("hidden_dim", 512))
        num_layers = int(model_cfg.get("num_layers", 2))
        dropout = float(model_cfg.get("dropout", 0.20))
        latent_channels = int(model_cfg.get("latent_channels", 256))
        latent_hw = int(model_cfg.get("latent_hw", 8))
        decoder_channels = list(model_cfg.get("decoder_channels", [256, 128, 64, 32]))
        decoder_dropout = float(model_cfg.get("decoder_dropout", 0.10))
        if decoder_channels[0] != latent_channels:
            raise ValueError("decoder_channels[0] must equal latent_channels.")

        self.embed = nn.Linear(1, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.pre_head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        latent_dim = latent_channels * latent_hw * latent_hw
        self.to_latent = nn.Linear(2 * hidden_dim, latent_dim)
        self.decoder = ConvDecoder(
            image_size=image_size,
            latent_hw=latent_hw,
            channels=decoder_channels,
            dropout=decoder_dropout,
        )
        self.latent_channels = latent_channels
        self.latent_hw = latent_hw
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x[:, : self.input_dim].unsqueeze(-1)
        seq = self.embed(seq)
        h, _ = self.lstm(seq)
        pooled = h.mean(dim=1)
        pooled = self.pre_head(pooled)
        z = self.to_latent(pooled)
        z = z.reshape(-1, self.latent_channels, self.latent_hw, self.latent_hw)
        return self.decoder(z)


class TransformerDecoderReconstructor(nn.Module):
    def __init__(self, input_dim: int, image_size: int, model_cfg: dict) -> None:
        super().__init__()
        token_dim = int(model_cfg.get("token_dim", 128))
        nhead = int(model_cfg.get("nhead", 8))
        num_layers = int(model_cfg.get("num_layers", 6))
        ff_dim = int(model_cfg.get("ff_dim", 512))
        dropout = float(model_cfg.get("dropout", 0.15))
        latent_channels = int(model_cfg.get("latent_channels", 256))
        latent_hw = int(model_cfg.get("latent_hw", 8))
        decoder_channels = list(model_cfg.get("decoder_channels", [256, 128, 64, 32]))
        decoder_dropout = float(model_cfg.get("decoder_dropout", 0.10))
        if decoder_channels[0] != latent_channels:
            raise ValueError("decoder_channels[0] must equal latent_channels.")

        self.token_proj = nn.Linear(1, token_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_dim, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(token_dim)

        latent_dim = latent_channels * latent_hw * latent_hw
        self.to_latent = nn.Linear(token_dim, latent_dim)
        self.decoder = ConvDecoder(
            image_size=image_size,
            latent_hw=latent_hw,
            channels=decoder_channels,
            dropout=decoder_dropout,
        )
        self.latent_channels = latent_channels
        self.latent_hw = latent_hw
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = x[:, : self.input_dim].unsqueeze(-1)
        tok = self.token_proj(tok)
        tok = tok + self.pos_embed[:, : tok.size(1), :]
        h = self.encoder(tok)
        pooled = self.norm(h.mean(dim=1))
        z = self.to_latent(pooled)
        z = z.reshape(-1, self.latent_channels, self.latent_hw, self.latent_hw)
        return self.decoder(z)


class EITReconstructor(nn.Module):
    """
    Model factory controlled by config keys:
      - model_name: str
      - model: dict   (active model parameters only)
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        input_dim = int(config["input_dim"])
        image_size = int(config["image_size"])
        model_name = str(config.get("model_name", "mlp")).lower()
        model_cfg = dict(config.get("model") or {})

        if model_name in {"mlp", "mlp_large", "mlp_xlarge"}:
            # Backward compatible path.
            if "hidden_dims" not in model_cfg and "hidden_dims" in config:
                model_cfg["hidden_dims"] = config["hidden_dims"]
            if "dropout" not in model_cfg and "dropout" in config:
                model_cfg["dropout"] = config["dropout"]
            self.net = LegacyMLPReconstructor(input_dim, image_size, model_cfg)
        elif model_name.startswith("res_mlp"):
            self.net = ResMLPReconstructor(input_dim, image_size, model_cfg)
        elif model_name.startswith("cnn_decoder"):
            self.net = CNNDecoderReconstructor(input_dim, image_size, model_cfg)
        elif model_name.startswith("bilstm_decoder"):
            self.net = BiLSTMDecoderReconstructor(input_dim, image_size, model_cfg)
        elif model_name.startswith("transformer_decoder"):
            self.net = TransformerDecoderReconstructor(input_dim, image_size, model_cfg)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        self.model_name = model_name
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
