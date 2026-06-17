"""HCAN model adapted to the unified forecast contract."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from windlab.models.base import flatten_airport_features, validate_forecast_input
from windlab.registry import MODELS


def _reshape_hierarchical_output(
    values: torch.Tensor,
    *,
    batch_size: int,
    forecast_steps: int,
    airport_count: int,
    target_size: int,
    class_count: int,
) -> torch.Tensor:
    return values.reshape(
        batch_size,
        forecast_steps,
        airport_count,
        target_size,
        class_count,
    )


class HCANModel(nn.Module):
    """HCAN with a GRU backbone adapted to the project forecast contract."""

    def __init__(
        self,
        *,
        input_size: int,
        input_steps: int,
        backbone_hidden_size: int,
        backbone_num_layers: int,
        backbone_dropout: float,
        hidden_dim: int,
        num_coarse: int,
        num_fine: int,
        lambda_cls: float,
        lambda_reg: float,
        lambda_acl: float,
        lambda_direct: float,
        forecast_steps: int,
        airport_count: int,
        target_size: int,
    ) -> None:
        super().__init__()
        if backbone_hidden_size <= 0:
            raise ValueError("backbone_hidden_size must be positive.")
        if backbone_num_layers <= 0:
            raise ValueError("backbone_num_layers must be positive.")
        if not 0.0 <= backbone_dropout < 1.0:
            raise ValueError("backbone_dropout must be in [0.0, 1.0).")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_coarse <= 0:
            raise ValueError("num_coarse must be positive.")
        if num_fine <= 0:
            raise ValueError("num_fine must be positive.")
        if num_fine != 2 * num_coarse:
            raise ValueError("num_fine must equal 2 * num_coarse.")
        for name, value in {
            "lambda_cls": lambda_cls,
            "lambda_reg": lambda_reg,
            "lambda_acl": lambda_acl,
            "lambda_direct": lambda_direct,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")

        self.input_size = input_size
        self.input_steps = input_steps
        self.backbone_hidden_size = backbone_hidden_size
        self.backbone_num_layers = backbone_num_layers
        self.backbone_dropout = backbone_dropout
        self.hidden_dim = hidden_dim
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.lambda_acl = lambda_acl
        self.lambda_direct = lambda_direct
        self.forecast_steps = forecast_steps
        self.airport_count = airport_count
        self.target_size = target_size
        self.channel_count = airport_count * target_size

        effective_dropout = backbone_dropout if backbone_num_layers > 1 else 0.0
        self.backbone = nn.GRU(
            input_size=input_size,
            hidden_size=backbone_hidden_size,
            num_layers=backbone_num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.backbone_projection = nn.Linear(
            backbone_hidden_size,
            forecast_steps * self.channel_count,
        )

        self.g_proj1 = nn.Linear(forecast_steps, hidden_dim)
        self.g_proj2 = nn.Linear(hidden_dim, forecast_steps)
        self.predictor = nn.Linear(forecast_steps, forecast_steps)

        self.c_proj1 = nn.Linear(forecast_steps, hidden_dim)
        self.coarse_predictor = nn.Linear(hidden_dim, forecast_steps * num_coarse)
        self.coarse_classify_layer = nn.Linear(hidden_dim, forecast_steps * num_coarse)

        self.f_proj1 = nn.Linear(forecast_steps, hidden_dim)
        self.fine_predictor = nn.Linear(hidden_dim, forecast_steps * num_fine)
        self.fine_classify_layer = nn.Linear(hidden_dim, forecast_steps * num_fine)

    @property
    def init_kwargs(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "input_steps": self.input_steps,
            "backbone_hidden_size": self.backbone_hidden_size,
            "backbone_num_layers": self.backbone_num_layers,
            "backbone_dropout": self.backbone_dropout,
            "hidden_dim": self.hidden_dim,
            "num_coarse": self.num_coarse,
            "num_fine": self.num_fine,
            "lambda_cls": self.lambda_cls,
            "lambda_reg": self.lambda_reg,
            "lambda_acl": self.lambda_acl,
            "lambda_direct": self.lambda_direct,
            "forecast_steps": self.forecast_steps,
            "airport_count": self.airport_count,
            "target_size": self.target_size,
        }

    def _pairwise_average(self, values: torch.Tensor) -> torch.Tensor:
        even = values[..., 0::2]
        odd = values[..., 1::2]
        return (even + odd) / 2.0

    def forward(self, inputs: torch.Tensor) -> dict[str, Any]:
        batch_size, _ = validate_forecast_input(inputs, self.input_size)
        flattened = flatten_airport_features(inputs)
        _, hidden = self.backbone(flattened)
        final_hidden = hidden[-1]
        initial_sequence = self.backbone_projection(final_hidden).reshape(
            batch_size,
            self.forecast_steps,
            self.channel_count,
        )
        x = initial_sequence.transpose(1, 2)

        coarse_features = self.c_proj1(x)
        coarse_prediction = self.coarse_predictor(coarse_features).reshape(
            batch_size,
            self.channel_count,
            self.forecast_steps,
            self.num_coarse,
        )
        coarse_logits = self.coarse_classify_layer(coarse_features).reshape(
            batch_size,
            self.channel_count,
            self.forecast_steps,
            self.num_coarse,
        )

        fine_features = self.f_proj1(x)
        fine_prediction = self.fine_predictor(fine_features).reshape(
            batch_size,
            self.channel_count,
            self.forecast_steps,
            self.num_fine,
        )
        fine_logits = self.fine_classify_layer(fine_features).reshape(
            batch_size,
            self.channel_count,
            self.forecast_steps,
            self.num_fine,
        )
        fine_logits_switch = self._pairwise_average(fine_logits)

        g_features = self.g_proj1(x)
        attention_scores = torch.matmul(coarse_features, fine_features.transpose(1, 2))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended = torch.matmul(attention_weights, g_features)
        residual = self.g_proj2(attended)
        direct_prediction = self.predictor(residual + x).transpose(1, 2)

        prediction = direct_prediction.reshape(
            batch_size,
            self.forecast_steps,
            self.airport_count,
            self.target_size,
        )
        coarse_prediction = _reshape_hierarchical_output(
            coarse_prediction.permute(0, 2, 1, 3).contiguous(),
            batch_size=batch_size,
            forecast_steps=self.forecast_steps,
            airport_count=self.airport_count,
            target_size=self.target_size,
            class_count=self.num_coarse,
        )
        coarse_logits = _reshape_hierarchical_output(
            coarse_logits.permute(0, 2, 1, 3).contiguous(),
            batch_size=batch_size,
            forecast_steps=self.forecast_steps,
            airport_count=self.airport_count,
            target_size=self.target_size,
            class_count=self.num_coarse,
        )
        fine_prediction = _reshape_hierarchical_output(
            fine_prediction.permute(0, 2, 1, 3).contiguous(),
            batch_size=batch_size,
            forecast_steps=self.forecast_steps,
            airport_count=self.airport_count,
            target_size=self.target_size,
            class_count=self.num_fine,
        )
        fine_logits = _reshape_hierarchical_output(
            fine_logits.permute(0, 2, 1, 3).contiguous(),
            batch_size=batch_size,
            forecast_steps=self.forecast_steps,
            airport_count=self.airport_count,
            target_size=self.target_size,
            class_count=self.num_fine,
        )
        fine_logits_switch = _reshape_hierarchical_output(
            fine_logits_switch.permute(0, 2, 1, 3).contiguous(),
            batch_size=batch_size,
            forecast_steps=self.forecast_steps,
            airport_count=self.airport_count,
            target_size=self.target_size,
            class_count=self.num_coarse,
        )
        return {
            "prediction": prediction,
            "aux": {
                "hidden_state": final_hidden,
                "hcan": {
                    "coarse_prediction": coarse_prediction,
                    "coarse_logits": coarse_logits,
                    "fine_prediction": fine_prediction,
                    "fine_logits": fine_logits,
                    "fine_logits_switch": fine_logits_switch,
                    "num_coarse": self.num_coarse,
                    "num_fine": self.num_fine,
                    "lambda_cls": self.lambda_cls,
                    "lambda_reg": self.lambda_reg,
                    "lambda_acl": self.lambda_acl,
                    "lambda_direct": self.lambda_direct,
                },
            },
        }


if "hcan" not in MODELS.keys():
    MODELS.register("hcan", HCANModel)
