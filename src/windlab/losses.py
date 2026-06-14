"""Loss functions used by the training pipeline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from windlab.registry import LOSSES

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def mse_loss(
    prediction: FloatArray,
    target: FloatArray,
    mask: BoolArray | None = None,
) -> float:
    errors = prediction - target
    squared = errors * errors
    if mask is None:
        return float(np.mean(squared))
    active = squared[mask]
    if active.size == 0:
        raise ValueError("Mask selects no elements for mse_loss.")
    return float(np.mean(active))


if "mse" not in LOSSES.keys():
    LOSSES.register("mse", mse_loss)
