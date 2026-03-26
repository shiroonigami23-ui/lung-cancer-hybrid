# Lung Cancer Hybrid Model (Sybil + ResNet3D)

Starter implementation of the Sybil-ResNet late-fusion blueprint from `Sybil_ResNet_Hybrid_Blueprint.pdf`.

## Current Status

- Project scaffold created from the blueprint module layout.
- Baseline hybrid model implemented:
  - global branch: `SybilFeatureExtractor` (real Sybil if installed, fallback 3D CNN)
  - local branch: `ResNet3DNodule` (3D encoder + 256-d embedding)
  - fusion head: malignancy logits + 6-year risk scores
- Minimal training/evaluation/inference pipeline included.

## Project Structure

```text
data/
models/
training/
evaluation/
inference/
notebooks/
train.py
requirements.txt
```

## Quick Start (Local or Colab)

```bash
pip install -r requirements.txt
python train.py --epochs 2 --batch-size 2 --num-workers 0 --mock-data
```

## Colab Notes

- Your notebook link is reachable.
- Use secrets already configured in Colab (`HF_TOKEN`, `WANDB_API_KEY`, etc.).
- Clone this repo in Colab and run training there for GPU acceleration.

## End-to-End Colab (Real Dataset + Pretrained + HF + App)

Run these cells in order:

```python
!git -C /content/lung-cancer-hybrid pull
%cd /content/lung-cancer-hybrid
!pip install -r requirements.txt
```

```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["HF_REPO_ID"] = "shiroonigami23-ui/lung-cancer-hybrid-nodulemnist3d"
```

```python
!python scripts/train_nodulemnist3d.py --epochs 8 --batch-size 32 --num-workers 2
!python scripts/push_to_hf.py --repo-id $HF_REPO_ID
```

```python
!python app/gradio_app.py --repo-id $HF_REPO_ID --share
```

Notes:
- Dataset: `NoduleMNIST3D` (MedMNIST, derived from LIDC-IDRI).
- Backbone: pretrained `r3d_18`, then fine-tuned.
- App input: upload a `.npy` 3D nodule patch.

## Next Build Steps

1. Plug real LIDC-IDRI parsing in `data/lidc_parser.py`.
2. Integrate official Sybil weights/token handling in `models/sybil_extractor.py`.
3. Add real validation on LIDC/NLST with calibration and reporting.
