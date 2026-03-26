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

## Next Build Steps

1. Plug real LIDC-IDRI parsing in `data/lidc_parser.py`.
2. Integrate official Sybil weights/token handling in `models/sybil_extractor.py`.
3. Add real validation on LIDC/NLST with calibration and reporting.

