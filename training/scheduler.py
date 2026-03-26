from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_warmup_cosine_scheduler(optimizer: Optimizer, warmup_epochs: int = 5, total_epochs: int = 60):
    cosine_epochs = max(1, total_epochs - warmup_epochs)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

