"""Metric functions with explicit mask support."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

from windlab.registry import METRICS

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
MetricFn = Callable[[FloatArray, FloatArray, BoolArray | None], float]


def _masked_view(
    prediction: FloatArray,
    target: FloatArray,
    mask: BoolArray | None,
) -> tuple[FloatArray, FloatArray]:
    if mask is None:
        return prediction.reshape(-1), target.reshape(-1)
    selected_prediction = prediction[mask]
    selected_target = target[mask]
    if selected_prediction.size == 0:
        raise ValueError("Mask selects no elements for metric computation.")
    return selected_prediction, selected_target


def mae_metric(
    prediction: FloatArray,
    target: FloatArray,
    mask: BoolArray | None = None,
) -> float:
    masked_prediction, masked_target = _masked_view(prediction, target, mask)
    return float(np.mean(np.abs(masked_prediction - masked_target)))


def rmse_metric(
    prediction: FloatArray,
    target: FloatArray,
    mask: BoolArray | None = None,
) -> float:
    masked_prediction, masked_target = _masked_view(prediction, target, mask)
    squared_error = (masked_prediction - masked_target) ** 2
    return float(np.sqrt(np.mean(squared_error)))


def bias_metric(
    prediction: FloatArray,
    target: FloatArray,
    mask: BoolArray | None = None,
) -> float:
    masked_prediction, masked_target = _masked_view(prediction, target, mask)
    return float(np.mean(masked_prediction - masked_target))


def compute_metrics(
    metric_names: list[str],
    prediction: FloatArray,
    target: FloatArray,
    mask: BoolArray | None,
) -> dict[str, float]:
    results: dict[str, float] = {}
    for metric_name in metric_names:
        metric_fn = cast(MetricFn, METRICS.get(metric_name))
        results[metric_name] = metric_fn(prediction, target, mask)
    return results


if "mae" not in METRICS.keys():
    METRICS.register("mae", mae_metric)
if "rmse" not in METRICS.keys():
    METRICS.register("rmse", rmse_metric)
if "bias" not in METRICS.keys():
    METRICS.register("bias", bias_metric)
