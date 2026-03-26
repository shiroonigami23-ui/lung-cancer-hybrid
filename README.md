# Lung Cancer Hybrid Model

Hybrid 3D lung nodule classification pipeline with a fusion architecture and deployment-ready inference app.

## Live Links

- Hugging Face Model: https://huggingface.co/ShiroOnigami23/lung-cancer-hybrid-model
- Hugging Face Space (Web App): https://huggingface.co/spaces/ShiroOnigami23/lung-cancer-hybrid-app

## Latest Results

- Test Accuracy: **84.19%**
- Test AUC: **0.9202**
- Test F1: **0.6918**
- Dataset used for benchmark run: **MedMNIST NoduleMNIST3D** (derived from LIDC-IDRI)

## Project Structure

```text
app.py
Dockerfile
requirements.txt
requirements-app.txt
data/
models/
training/
evaluation/
inference/
scripts/
```

## Recommended Framework

For fastest iteration + shareable demo, use:

- **Gradio + Hugging Face Spaces** (already set up)

Reason: easiest deployment flow, direct model hosting, and instant public/private demos with zero custom backend maintenance.

## Run App Locally

```bash
pip install -r requirements-app.txt
python app.py
```

Then open `http://localhost:7860`.

## Run With Docker

```bash
docker build -t lung-cancer-hybrid .
docker run -p 7860:7860 lung-cancer-hybrid
```

## Train

```bash
python train.py --epochs 2 --batch-size 2 --num-workers 0 --mock-data
```

For real-dataset training in Colab, use scripts under `scripts/`.

## Notes

- The app expects a `.npy` 3D nodule patch input.
- `app.py` loads weights directly from the Hugging Face model repo.

