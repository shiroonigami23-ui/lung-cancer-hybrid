from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    out: Dict[str, float] = {}
    out["f1"] = float(f1_score(y_true, y_pred))
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["auc"] = float("nan")
    return out

