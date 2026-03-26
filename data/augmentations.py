from __future__ import annotations

from typing import Callable

try:
    from monai.transforms import Compose, RandFlipd, RandRotate90d, RandScaleIntensityd
except Exception:
    Compose = None  # type: ignore[assignment]


def build_train_transforms() -> Callable | None:
    if Compose is None:
        return None
    return Compose(
        [
            RandFlipd(keys=["volume", "patch"], prob=0.5, spatial_axis=[0]),
            RandRotate90d(keys=["volume", "patch"], prob=0.3, max_k=3),
            RandScaleIntensityd(keys=["volume", "patch"], factors=0.1, prob=0.5),
        ]
    )

