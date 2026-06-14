"""DLinear baseline adapted to the forecast contract."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional

from windlab.models.base import (
    flatten_airport_features,
    reshape_prediction,
    validate_forecast_input,
)
from windlab.registry import MODELS


class DLinearModel(nn.Module):
    """Decomposition-linear model with trend and seasonal linear heads."""

    def __init__(
        self,
        *,
        input_size: int,
        input_steps: int,
        forecast_steps: int,
        airport_count: int,
        target_size: int,
        moving_avg: int,
        individual: bool,
    ) -> None:
        super().__init__()
        if moving_avg <= 0 or moving_avg % 2 == 0:
            raise ValueError("moving_avg must be a positive odd integer.")
        self.input_size = input_size
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.airport_count = airport_count
        self.target_size = target_size
        self.moving_avg = moving_avg
        self.individual = individual

        if individual:
            self.seasonal_layers = nn.ModuleList(
                nn.Linear(input_steps, forecast_steps) for _ in range(input_size)
            )
            self.trend_layers = nn.ModuleList(
                nn.Linear(input_steps, forecast_steps) for _ in range(input_size)
            )
        else:
            self.seasonal_layer = nn.Linear(input_steps, forecast_steps)
            self.trend_layer = nn.Linear(input_steps, forecast_steps)
        self.output_projection = nn.Linear(input_size, airport_count * target_size)

    @property
    def init_kwargs(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "input_steps": self.input_steps,
            "forecast_steps": self.forecast_steps,
            "airport_count": self.airport_count,
            "target_size": self.target_size,
            "moving_avg": self.moving_avg,
            "individual": self.individual,
        }

    def forward(self, inputs: torch.Tensor) -> dict[str, Any]:
        batch_size, _ = validate_forecast_input(inputs, self.input_size)
        flattened = flatten_airport_features(inputs)
        trend = self._moving_average(flattened)
        seasonal = flattened - trend
        forecast = self._linear_forecast(seasonal, trend)
        projected = self.output_projection(forecast)
        prediction = reshape_prediction(
            projected,
            batch_size=batch_size,
            forecast_steps=self.forecast_steps,
            airport_count=self.airport_count,
            target_size=self.target_size,
        )
        return {"prediction": prediction, "aux": {"trend": trend, "seasonal": seasonal}}

    def _moving_average(self, values: torch.Tensor) -> torch.Tensor:
        padding = self.moving_avg // 2
        padded = functional.pad(
            values.transpose(1, 2),
            (padding, padding),
            mode="replicate",
        )
        averaged = functional.avg_pool1d(padded, kernel_size=self.moving_avg, stride=1)
        return averaged.transpose(1, 2)

    def _linear_forecast(
        self,
        seasonal: torch.Tensor,
        trend: torch.Tensor,
    ) -> torch.Tensor:
        if not self.individual:
            seasonal_out = self.seasonal_layer(seasonal.transpose(1, 2))
            trend_out = self.trend_layer(trend.transpose(1, 2))
            return cast(torch.Tensor, (seasonal_out + trend_out).transpose(1, 2))

        channel_outputs: list[torch.Tensor] = []
        for channel in range(self.input_size):
            seasonal_channel = self.seasonal_layers[channel](seasonal[:, :, channel])
            trend_channel = self.trend_layers[channel](trend[:, :, channel])
            channel_outputs.append(seasonal_channel + trend_channel)
        return torch.stack(channel_outputs, dim=-1)


if "dlinear" not in MODELS.keys():
    MODELS.register("dlinear", DLinearModel)
