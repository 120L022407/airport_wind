"""PyTorch GRU baseline model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from windlab.registry import MODELS


class GRUModel(nn.Module):
    """Encode observed history with `torch.nn.GRU` and predict all horizons."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        forecast_steps: int,
        airport_count: int,
        target_size: int,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.forecast_steps = forecast_steps
        self.airport_count = airport_count
        self.target_size = target_size

        effective_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.projection = nn.Linear(
            hidden_size,
            forecast_steps * airport_count * target_size,
        )

    @property
    def init_kwargs(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "forecast_steps": self.forecast_steps,
            "airport_count": self.airport_count,
            "target_size": self.target_size,
        }

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4:
            raise ValueError(
                "GRUModel inputs must have shape "
                "[batch, input_steps, airport, feature]."
            )
        batch_size, input_steps, airport_count, feature_count = inputs.shape
        expected_input_size = airport_count * feature_count
        if expected_input_size != self.input_size:
            raise ValueError(
                f"Expected flattened input size {self.input_size}, "
                f"got {expected_input_size}."
            )

        flattened = inputs.reshape(batch_size, input_steps, self.input_size)
        _, hidden = self.encoder(flattened)
        final_hidden = hidden[-1]
        prediction = self.projection(final_hidden).reshape(
            batch_size,
            self.forecast_steps,
            self.airport_count,
            self.target_size,
        )
        return {"prediction": prediction, "aux": {"hidden_state": final_hidden}}


if "gru" not in MODELS.keys():
    MODELS.register("gru", GRUModel)
