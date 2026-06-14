from __future__ import annotations

import numpy as np

from windlab.metrics import compute_metrics


def test_metrics_use_masked_observations_only() -> None:
    prediction = np.array([[[[1.0]], [[100.0]]]])
    target = np.array([[[[3.0]], [[0.0]]]])
    mask = np.array([[[[True]], [[False]]]])
    metrics = compute_metrics(["mae", "rmse", "bias"], prediction, target, mask)
    assert metrics["mae"] == 2.0
    assert metrics["rmse"] == 2.0
    assert metrics["bias"] == -2.0
