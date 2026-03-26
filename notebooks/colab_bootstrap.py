from google.colab import userdata

import os
import subprocess
from pathlib import Path


REPO_URL = "https://github.com/shiroonigami23-ui/lung-cancer-hybrid.git"
REPO_DIR = Path("/content/lung-cancer-hybrid")


def run(cmd: str) -> None:
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def maybe_set_env(secret_name: str) -> None:
    try:
        value = userdata.get(secret_name)
        if value:
            os.environ[secret_name] = value
            print(f"Loaded secret: {secret_name}")
    except Exception:
        print(f"Secret not found or unavailable: {secret_name}")


if REPO_DIR.exists():
    run(f"cd {REPO_DIR} && git pull")
else:
    run(f"git clone {REPO_URL} {REPO_DIR}")

run(f"cd {REPO_DIR} && pip install -r requirements.txt")

for name in ["HF_TOKEN", "WANDB_API_KEY", "KAGGLE_KEY", "GITHUB_TOKEN"]:
    maybe_set_env(name)

run(f"cd {REPO_DIR} && python train.py --epochs 2 --batch-size 2 --num-workers 2 --mock-data")
print("Colab bootstrap complete.")

