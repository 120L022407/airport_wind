"""Shared helpers for forecast models."""

from __future__ import annotations

import torch


def validate_forecast_input(inputs: torch.Tensor, input_size: int) -> tuple[int, int]:
    if inputs.ndim != 4:
        raise ValueError(
            "Forecast model inputs must have shape "
            "[batch, input_steps, airport, feature]."
        )
    batch_size, input_steps, airport_count, feature_count = inputs.shape
    flattened_size = airport_count * feature_count
    if flattened_size != input_size:
        raise ValueError(f"Expected flattened input size {input_size}, got {flattened_size}.")
    return batch_size, input_steps


def flatten_airport_features(inputs: torch.Tensor) -> torch.Tensor:
    batch_size, input_steps, airport_count, feature_count = inputs.shape
    return inputs.reshape(batch_size, input_steps, airport_count * feature_count)


def reshape_prediction(
    values: torch.Tensor,
    *,
    batch_size: int,
    forecast_steps: int,
    airport_count: int,
    target_size: int,
) -> torch.Tensor:
    return values.reshape(batch_size, forecast_steps, airport_count, target_size)
