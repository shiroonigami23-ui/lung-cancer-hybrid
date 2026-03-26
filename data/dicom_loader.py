from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


def load_dicom_volume(dicom_dir: str | Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not files:
        raise FileNotFoundError(f"No DICOM series found in: {dicom_dir}")

    reader.SetFileNames(files)
    image = reader.Execute()

    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    volume = np.clip(volume, -1000.0, 400.0)
    volume = (volume + 1000.0) / 1400.0
    spacing = image.GetSpacing()
    return volume, spacing


def resample_to_isotropic(
    volume: np.ndarray, orig_spacing: Tuple[float, float, float], target_mm: float = 1.0
) -> np.ndarray:
    scale = [s / target_mm for s in orig_spacing]
    return zoom(volume, scale, order=1)

