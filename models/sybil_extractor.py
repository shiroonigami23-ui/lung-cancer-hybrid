from __future__ import annotations

import torch
import torch.nn as nn


class _FallbackGlobalEncoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, out_dim, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        x = self.encoder(volume)
        return self.pool(x).flatten(1)


class SybilFeatureExtractor(nn.Module):
    def __init__(self, use_pretrained_sybil: bool = False, freeze: bool = True):
        super().__init__()
        self.backend_name = "fallback"
        self.model: nn.Module

        if use_pretrained_sybil:
            try:
                from sybil import Sybil  # type: ignore

                sybil = Sybil("sybil_1")
                self.model = nn.Sequential(sybil.encoder, sybil.pool)
                self.backend_name = "sybil"
            except Exception:
                self.model = _FallbackGlobalEncoder(out_dim=512)
        else:
            self.model = _FallbackGlobalEncoder(out_dim=512)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        out = self.model(volume)
        return out.flatten(1)

