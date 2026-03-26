from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", type=str, required=True, help="e.g. shiroonigami23-ui/lung-cancer-hybrid-nodulemnist3d")
    p.add_argument("--artifact-dir", type=str, default="./artifacts/nodulemnist3d")
    p.add_argument("--private", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN or HUGGINGFACE_TOKEN env var is required.")

    artifact_dir = Path(args.artifact_dir)
    model_path = artifact_dir / "best_model.pt"
    metrics_path = artifact_dir / "metrics.json"
    history_path = artifact_dir / "history.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    create_repo(repo_id=args.repo_id, token=token, private=args.private, exist_ok=True)
    api = HfApi(token=token)

    card = {
        "model_type": "r3d_18_finetuned_on_nodulemnist3d",
        "task": "lung nodule malignancy classification",
        "dataset": "medmnist/nodulemnist3d",
    }
    (artifact_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    api.upload_file(path_or_fileobj=str(model_path), path_in_repo="best_model.pt", repo_id=args.repo_id)
    if metrics_path.exists():
        api.upload_file(path_or_fileobj=str(metrics_path), path_in_repo="metrics.json", repo_id=args.repo_id)
    if history_path.exists():
        api.upload_file(path_or_fileobj=str(history_path), path_in_repo="history.json", repo_id=args.repo_id)
    api.upload_file(
        path_or_fileobj=str(artifact_dir / "model_card.json"),
        path_in_repo="model_card.json",
        repo_id=args.repo_id,
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

