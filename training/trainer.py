from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from training.losses import total_loss


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        volume = batch["volume"].to(device)
        patch = batch["patch"].to(device)
        labels = batch["label"].to(device)
        risk_labels = batch["risk_labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        mal_logits, risk_preds = model(volume, patch)
        loss = total_loss(mal_logits, risk_preds, labels, risk_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


@torch.no_grad()
def validate_epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0.0
    for batch in loader:
        volume = batch["volume"].to(device)
        patch = batch["patch"].to(device)
        labels = batch["label"].to(device)
        risk_labels = batch["risk_labels"].to(device)
        mal_logits, risk_preds = model(volume, patch)
        total += float(total_loss(mal_logits, risk_preds, labels, risk_labels).item())
    return {"val_loss": total / max(1, len(loader))}

