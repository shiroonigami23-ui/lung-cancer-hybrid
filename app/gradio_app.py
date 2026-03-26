from __future__ import annotations

import argparse
from typing import Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


def build_model() -> nn.Module:
    model = r3d_18(weights=None)
    model.stem[0] = nn.Conv3d(
        1,
        64,
        kernel_size=(3, 7, 7),
        stride=(1, 2, 2),
        padding=(1, 3, 3),
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def load_model(repo_id: str) -> Tuple[nn.Module, torch.device]:
    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download(repo_id=repo_id, filename="best_model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, device


def preprocess(arr: np.ndarray) -> torch.Tensor:
    # Input expected: npy with shape (D,H,W) around lung nodule region.
    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x = torch.nn.functional.interpolate(x, size=(64, 64, 64), mode="trilinear", align_corners=False)
    x = x / (x.max() + 1e-6)
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    model, device = load_model(args.repo_id)

    def predict(npy_file):
        arr = np.load(npy_file.name)
        x = preprocess(arr).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
        verdict = "MALIGNANT" if prob >= 0.5 else "BENIGN"
        return {"benign": round(1 - prob, 4), "malignant": round(prob, 4)}, verdict

    demo = gr.Interface(
        fn=predict,
        inputs=gr.File(label="Upload 3D Nodule .npy"),
        outputs=[gr.Label(label="Probabilities"), gr.Textbox(label="Verdict")],
        title="Lung Nodule Malignancy Classifier",
        description="Upload a 3D nodule patch as .npy array (D,H,W).",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()

