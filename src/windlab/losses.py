"""Loss functions used by the training pipeline."""

from __future__ import annotations

import torch

from windlab.registry import LOSSES


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    squared_error = (prediction - target) ** 2
    if mask is None:
        return squared_error.mean()
    active = squared_error[mask.bool()]
    if active.numel() == 0:
        raise ValueError("Mask selects no elements for masked_mse_loss.")
    return active.mean()


if "mse" not in LOSSES.keys():
    LOSSES.register("mse", masked_mse_loss)
