from __future__ import annotations

import os
import subprocess


def run(cmd: str) -> None:
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    repo_id = os.environ.get("HF_REPO_ID", "").strip()
    if not repo_id:
        raise RuntimeError("Set HF_REPO_ID, e.g. export HF_REPO_ID=shiroonigami23-ui/lung-cancer-hybrid-nodulemnist3d")

    run("python scripts/train_nodulemnist3d.py --epochs 8 --batch-size 32 --num-workers 2")
    run(f"python scripts/push_to_hf.py --repo-id {repo_id}")
    print("Training + HF upload done.")


if __name__ == "__main__":
    main()

