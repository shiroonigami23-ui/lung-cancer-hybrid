from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision.models.video import R3D_18_Weights, r3d_18
from tqdm import tqdm

import medmnist
from medmnist import INFO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data_raw")
    p.add_argument("--outdir", type=str, default="./artifacts/nodulemnist3d")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _upsample_3d_to_64(x: torch.Tensor) -> torch.Tensor:
    # Input from NoduleMNIST3D is usually (B, 1, 28, 28, 28). Upsample for r3d_18.
    return torch.nn.functional.interpolate(x, size=(64, 64, 64), mode="trilinear", align_corners=False)


def _prepare_batch_x(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.float()
    # Handle possible shapes from MedMNIST:
    # (B, D, H, W) or (B, 1, D, H, W) or (B, D, H, W, 1)
    if x.ndim == 4:
        x = x.unsqueeze(1)
    elif x.ndim == 5 and x.shape[-1] == 1:
        x = x.permute(0, 4, 1, 2, 3).contiguous()
    elif x.ndim == 5 and x.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")

    x = x.to(device)
    x = x / 255.0 if x.max() > 1.0 else x
    return _upsample_3d_to_64(x)


def build_model(num_classes: int = 2) -> nn.Module:
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    model.stem[0] = nn.Conv3d(
        1,
        64,
        kernel_size=(3, 7, 7),
        stride=(1, 2, 2),
        padding=(1, 3, 3),
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = _prepare_batch_x(x, device)
            y = y.squeeze(1).long().to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            probs_all.append(probs.detach().cpu().numpy())
            labels_all.append(y.detach().cpu().numpy())

    probs_np = np.concatenate(probs_all)
    labels_np = np.concatenate(labels_all)
    preds_np = (probs_np >= 0.5).astype(np.int64)
    acc = accuracy_score(labels_np, preds_np)
    auc = roc_auc_score(labels_np, probs_np)
    return float(acc), float(auc)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(args.data_root).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    info = INFO["nodulemnist3d"]
    DataClass = getattr(medmnist, info["python_class"])

    train_ds = DataClass(split="train", root=args.data_root, download=True)
    val_ds = DataClass(split="val", root=args.data_root, download=True)
    test_ds = DataClass(split="test", root=args.data_root, download=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_auc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        loop = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for x, y in loop:
            x = _prepare_batch_x(x, device)
            y = y.squeeze(1).long().to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
            loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        val_acc, val_auc = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_acc": val_acc,
            "val_auc": val_auc,
        }
        history.append(row)
        print(row)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Path(args.outdir) / "best_model.pt")

    model.load_state_dict(torch.load(Path(args.outdir) / "best_model.pt", map_location=device))
    test_acc, test_auc = evaluate(model, test_loader, device)
    final_metrics = {"best_val_auc": best_auc, "test_acc": test_acc, "test_auc": test_auc}
    print("final_metrics", final_metrics)

    with open(Path(args.outdir) / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(Path(args.outdir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)


if __name__ == "__main__":
    main()
