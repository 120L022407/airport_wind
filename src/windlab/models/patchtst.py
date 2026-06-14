"""PatchTST baseline adapted to the forecast contract."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from windlab.models.base import flatten_airport_features, reshape_prediction
from windlab.models.base import validate_forecast_input
from windlab.registry import MODELS


class PatchTSTModel(nn.Module):
    """Channel-independent patch Transformer for fixed-horizon forecasting."""

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
        patch_len: int,
        stride: int,
    ) -> None:
        super().__init__()
        if patch_len > input_steps:
            raise ValueError("patch_len must be <= input_steps.")
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
        self.patch_len = patch_len
        self.stride = stride
        self.patch_count = ((input_steps - patch_len) // stride) + 1

        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 1, self.patch_count, d_model)
        )
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
        self.horizon_head = nn.Linear(self.patch_count * d_model, forecast_steps)
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
            "patch_len": self.patch_len,
            "stride": self.stride,
        }

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, _ = validate_forecast_input(inputs, self.input_size)
        flattened = flatten_airport_features(inputs).transpose(1, 2)
        patches = flattened.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        embedded = self.patch_embedding(patches) + self.position_embedding
        channel_independent_tokens = embedded.reshape(
            batch_size * self.input_size,
            self.patch_count,
            self.d_model,
        )
        encoded = self.encoder(channel_independent_tokens)
        encoded = encoded.reshape(
            batch_size,
            self.input_size,
            self.patch_count * self.d_model,
        )
        per_variate_forecast = self.horizon_head(encoded).transpose(1, 2)
        projected = self.output_projection(per_variate_forecast)
        prediction = reshape_prediction(
            projected,
            batch_size=batch_size,
            forecast_steps=self.forecast_steps,
            airport_count=self.airport_count,
            target_size=self.target_size,
        )
        return {"prediction": prediction, "aux": {"encoded": encoded}}


if "patchtst" not in MODELS.keys():
    MODELS.register("patchtst", PatchTSTModel)
