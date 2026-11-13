# 🗺 Roadmap — Lung Cancer Hybrid Model

> Back to the [README](../README.md).

## Where the project stands

Status: **live**.

## Milestones

### v1 — Hybrid baseline (done)

- [x] Hybrid architecture with fusion MLP
- [x] NoduleMNIST3D benchmark (AUC 0.92)
- [x] Grad-CAM, report generation, Gradio Space, Docker

### v1.1 — Stronger evaluation

*Goal:* Move from MedMNIST to full-resolution data.

- [ ] Full LIDC-IDRI patient-level cross-validation
- [ ] NLST longitudinal evaluation (multi-year risk)
- [ ] Calibration and decision-curve analysis

### v1.2 — Model

- [ ] Improve F1 with focal loss / class-balanced sampling
- [ ] Self-supervised pre-training on unlabeled CT
- [ ] Uncertainty estimates (MC dropout / ensembles)

### v2.0

- [ ] End-to-end detection + classification on raw DICOM in the app
- [ ] Radiologist-in-the-loop review UI

## 🔭 Long-term Vision

Accurate, explainable lung-nodule triage that helps radiologists prioritise the scans that matter most.

## How to help

Pick any unchecked item, open an issue to discuss the approach, then send a PR. Items in the earliest unfinished milestone are the highest priority.


---
*Last updated: 2026-11-13*
