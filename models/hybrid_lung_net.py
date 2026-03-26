from __future__ import annotations

import torch
import torch.nn as nn

from models.fusion_mlp import FusionMLP


class HybridLungNet(nn.Module):
    def __init__(self, sybil_extractor: nn.Module, resnet_nodule: nn.Module, fusion_dim: int = 384):
        super().__init__()
        self.global_branch = sybil_extractor
        self.local_branch = resnet_nodule
        self.fusion_mlp = FusionMLP(in_dim=512 + 256, fusion_dim=fusion_dim)

        hidden = fusion_dim // 2
        self.malignancy_head = nn.Linear(hidden, 2)
        self.risk_head = nn.Sequential(nn.Linear(hidden, 6), nn.Sigmoid())

    def forward(self, volume: torch.Tensor, patch: torch.Tensor):
        g_embed = self.global_branch(volume)
        l_embed = self.local_branch(patch, return_embed=True)
        fused = torch.cat([g_embed, l_embed], dim=1)
        x = self.fusion_mlp(fused)
        return self.malignancy_head(x), self.risk_head(x)

