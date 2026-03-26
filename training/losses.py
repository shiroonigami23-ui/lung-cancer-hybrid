from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCELoss(nn.Module):
    def __init__(self, pos_weight: float = 3.0):
        super().__init__()
        w = torch.tensor([1.0, pos_weight], dtype=torch.float32)
        self.register_buffer("weights", w)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels, weight=self.weights.to(logits.device))


def total_loss(
    mal_logits: torch.Tensor,
    risk_preds: torch.Tensor,
    mal_labels: torch.Tensor,
    risk_labels: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> torch.Tensor:
    mal_loss = WeightedCELoss()(mal_logits, mal_labels)
    risk_loss = nn.BCELoss()(risk_preds, risk_labels.float())
    return alpha * mal_loss + beta * risk_loss
