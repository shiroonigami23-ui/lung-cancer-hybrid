from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from data.dicom_loader import load_dicom_volume, resample_to_isotropic
from data.lidc_parser import extract_nodule_patch


@torch.no_grad()
def predict_patient(
    dicom_dir: str,
    nodule_coords_list: Sequence[Tuple[int, int, int]],
    model: torch.nn.Module,
    device: torch.device,
) -> List[Dict]:
    volume, spacing = load_dicom_volume(dicom_dir)
    volume_iso = resample_to_isotropic(volume, spacing)
    volume_t = torch.from_numpy(volume_iso).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    results = []
    for coord in nodule_coords_list:
        patch = extract_nodule_patch(volume_iso, coord, 64)
        patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
        mal_logits, risk_scores = model(volume_t, patch_t)
        mal_prob = torch.softmax(mal_logits, dim=1)[0, 1].item()
        risk = risk_scores[0].detach().cpu().tolist()
        results.append(
            {
                "nodule_coord": coord,
                "malignancy_prob": round(mal_prob, 4),
                "risk_yr1_to_yr6": [round(float(r), 4) for r in risk],
                "verdict": "MALIGNANT" if mal_prob > 0.5 else "BENIGN",
            }
        )
    return results

