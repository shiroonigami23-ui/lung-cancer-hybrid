from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def extract_nodule_patch(volume: np.ndarray, coord_zyx: Sequence[int], patch_size: int = 64) -> np.ndarray:
    z, y, x = [int(v) for v in coord_zyx]
    half = patch_size // 2

    zs, ze = max(0, z - half), min(volume.shape[0], z + half)
    ys, ye = max(0, y - half), min(volume.shape[1], y + half)
    xs, xe = max(0, x - half), min(volume.shape[2], x + half)

    patch = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
    cropped = volume[zs:ze, ys:ye, xs:xe]
    patch[: cropped.shape[0], : cropped.shape[1], : cropped.shape[2]] = cropped
    return patch


@dataclass
class LIDCRecord:
    volume_path: str
    nodule_coord_zyx: Tuple[int, int, int]
    malignancy_label: int
    risk_label_6y: Tuple[float, float, float, float, float, float]


class LIDCDataset(Dataset):
    def __init__(self, records: List[LIDCRecord], mock_data: bool = False):
        self.records = records
        self.mock_data = mock_data

    def __len__(self) -> int:
        return max(64, len(self.records)) if self.mock_data else len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mock_data or not self.records:
            volume = torch.rand(1, 96, 96, 96, dtype=torch.float32)
            patch = torch.rand(1, 64, 64, 64, dtype=torch.float32)
            label = torch.randint(0, 2, (1,), dtype=torch.long).squeeze(0)
            risk = torch.rand(6, dtype=torch.float32)
            return {"volume": volume, "patch": patch, "label": label, "risk_labels": risk}

        rec = self.records[idx]
        arr = np.load(rec.volume_path).astype(np.float32)
        patch = extract_nodule_patch(arr, rec.nodule_coord_zyx, 64)
        return {
            "volume": torch.from_numpy(arr).unsqueeze(0),
            "patch": torch.from_numpy(patch).unsqueeze(0),
            "label": torch.tensor(rec.malignancy_label, dtype=torch.long),
            "risk_labels": torch.tensor(rec.risk_label_6y, dtype=torch.float32),
        }


def build_lidc_dataset(data_root: str | Path = "./data_processed/lidc", mock_data: bool = True) -> LIDCDataset:
    if mock_data:
        return LIDCDataset(records=[], mock_data=True)

    data_root = Path(data_root)
    records: List[LIDCRecord] = []
    meta_path = data_root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Expected metadata file at {meta_path}")

    # Real parser can be added after dataset ingestion workflow is finalized.
    return LIDCDataset(records=records, mock_data=False)

