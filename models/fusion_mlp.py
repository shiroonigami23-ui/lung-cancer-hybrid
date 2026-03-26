from __future__ import annotations

import torch.nn as nn


class FusionMLP(nn.Module):
    def __init__(self, in_dim: int = 768, fusion_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.net(x)

