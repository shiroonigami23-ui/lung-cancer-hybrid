from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data.lidc_parser import build_lidc_dataset
from models.hybrid_lung_net import HybridLungNet
from models.resnet3d import ResNet3DNodule
from models.sybil_extractor import SybilFeatureExtractor
from training.scheduler import build_warmup_cosine_scheduler
from training.trainer import train_epoch, validate_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mock-data", action="store_true")
    p.add_argument("--use-pretrained-sybil", action="store_true")
    p.add_argument("--outdir", type=str, default="checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    dataset = build_lidc_dataset(mock_data=args.mock_data)
    n_val = max(1, len(dataset) // 10)
    n_train = max(1, len(dataset) - n_val)
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sybil_extractor = SybilFeatureExtractor(use_pretrained_sybil=args.use_pretrained_sybil, freeze=False)
    local_branch = ResNet3DNodule(embed_dim=256, num_classes=2)
    model = HybridLungNet(sybil_extractor, local_branch).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_epochs=min(2, args.epochs), total_epochs=args.epochs)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_stats = validate_epoch(model, val_loader, device)
        scheduler.step()
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_stats['val_loss']:.4f}")

        if val_stats["val_loss"] < best:
            best = val_stats["val_loss"]
            torch.save(model.state_dict(), outdir / "best_model.pt")

    print(f"done. best_val_loss={best:.4f}")


if __name__ == "__main__":
    main()

