"""iTransformer baseline adapted to the forecast contract."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from windlab.models.base import (
    flatten_airport_features,
    reshape_prediction,
    validate_forecast_input,
)
from windlab.registry import MODELS


class ITransformerModel(nn.Module):
    """Inverted Transformer with variates as tokens."""

    def __init__(
        self,
        *,
        input_size: int,
        input_steps: int,
        forecast_steps: int,
        airport_count: int,
        target_size: int,
        d_model: int,
        num_layers: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.airport_count = airport_count
        self.target_size = target_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.value_embedding = nn.Linear(input_steps, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.horizon_projection = nn.Linear(d_model, forecast_steps)
        self.output_projection = nn.Linear(input_size, airport_count * target_size)

    @property
    def init_kwargs(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "input_steps": self.input_steps,
            "forecast_steps": self.forecast_steps,
            "airport_count": self.airport_count,
            "target_size": self.target_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "n_heads": self.n_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        }

    def forward(self, inputs: torch.Tensor) -> dict[str, Any]:
        batch_size, _ = validate_forecast_input(inputs, self.input_size)
        flattened = flatten_airport_features(inputs)
        variate_tokens = flattened.transpose(1, 2)
        embedded = self.value_embedding(variate_tokens)
        encoded = self.encoder(embedded)
        per_variate_forecast = self.horizon_projection(encoded).transpose(1, 2)
        projected = self.output_projection(per_variate_forecast)
        prediction = reshape_prediction(
            projected,
            batch_size=batch_size,
            forecast_steps=self.forecast_steps,
            airport_count=self.airport_count,
            target_size=self.target_size,
        )
        return {"prediction": prediction, "aux": {"encoded": encoded}}


if "itransformer" not in MODELS.keys():
    MODELS.register("itransformer", ITransformerModel)
