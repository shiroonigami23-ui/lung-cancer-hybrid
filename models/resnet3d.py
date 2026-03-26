from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


class ResNet3DNodule(nn.Module):
    def __init__(self, embed_dim: int = 256, num_classes: int = 2):
        super().__init__()
        base = r3d_18(weights=None)
        base.stem[0] = nn.Conv3d(
            1,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )

        self.backbone = nn.Sequential(
            base.stem,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            nn.AdaptiveAvgPool3d(1),
        )
        self.embed_head = nn.Sequential(nn.Flatten(), nn.Linear(512, embed_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(embed_dim, num_classes))

    def forward(self, patch: torch.Tensor, return_embed: bool = False):
        x = self.backbone(patch)
        embed = self.embed_head(x)
        if return_embed:
            return embed
        logits = self.classifier(embed)
        return logits

