import os
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torchvision.models.video import r3d_18


MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "ShiroOnigami23/lung-cancer-hybrid-model")
CKPT_NAME = os.getenv("CKPT_NAME", "best.pt")


class Branch3D(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        base = r3d_18(weights=None)
        base.stem[0] = nn.Conv3d(
            1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )
        self.features = nn.Sequential(
            base.stem,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(x))


class HybridClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_branch = Branch3D(256)
        self.local_branch = Branch3D(256)
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.cls_head = nn.Linear(128, 2)

    def forward(self, g: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        ge = self.global_branch(g)
        le = self.local_branch(p)
        z = self.fusion(torch.cat([ge, le], dim=1))
        return self.cls_head(z)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = HybridClassifier().to(DEVICE)
MODEL_STATUS = "ok"

try:
    ckpt_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=CKPT_NAME)
    state = torch.load(ckpt_path, map_location=DEVICE)
    MODEL.load_state_dict(state)
    MODEL.eval()
except Exception as exc:
    MODEL_STATUS = f"error: {exc}"


def _prep(arr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(arr, dtype=torch.float32)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    elif x.ndim == 4 and x.shape[-1] == 1:
        x = x.permute(3, 0, 1, 2)
    elif x.ndim == 4 and x.shape[0] == 1:
        pass
    else:
        raise ValueError(f"Unsupported shape {tuple(x.shape)}. Expected (D,H,W) or (1,D,H,W).")

    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    g = F.interpolate(x.unsqueeze(0), size=(64, 64, 64), mode="trilinear", align_corners=False)
    p = g.clone()
    return g.to(DEVICE), p.to(DEVICE)


def predict(npy_file):
    if MODEL_STATUS != "ok":
        return {"error": 1.0}, f"Model load failed: {MODEL_STATUS}"

    arr = np.load(npy_file.name)
    g, p = _prep(arr)
    with torch.no_grad():
        prob = torch.softmax(MODEL(g, p), dim=1)[0, 1].item()
    return {"benign": round(1 - prob, 4), "malignant": round(prob, 4)}, (
        "MALIGNANT" if prob >= 0.5 else "BENIGN"
    )


demo = gr.Interface(
    fn=predict,
    inputs=gr.File(label="Upload 3D Nodule .npy"),
    outputs=[gr.Label(label="Class Probabilities"), gr.Textbox(label="Verdict")],
    title="Lung Cancer Hybrid Classifier",
    description=f"Loads weights from `{MODEL_REPO_ID}` and predicts benign vs malignant from 3D nodule patches.",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))

