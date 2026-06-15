"""TimeBridge model adapted to the forecast contract."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional

from windlab.models.base import reshape_prediction, validate_forecast_input
from windlab.registry import MODELS

ActivationFn = Callable[[torch.Tensor], torch.Tensor]


def _period_norm(values: torch.Tensor, stable_len: int) -> torch.Tensor:
    """Replicate the released TimeBridge patch normalization behavior."""

    if values.ndim == 3:
        values = values.unsqueeze(-2)
    batch_size, channel_count, patch_count, width = values.shape
    effective_len = min(stable_len, width)
    patches = [
        values[..., effective_len - 1 - offset : width - offset]
        for offset in range(effective_len)
    ]
    stacked = torch.stack(patches, dim=-1)
    mean = stacked.mean(dim=-1)
    padded = functional.pad(
        mean.reshape(batch_size * channel_count, patch_count, -1),
        pad=(effective_len - 1, 0),
        mode="replicate",
    )
    restored = padded.reshape(batch_size, channel_count, patch_count, -1)
    return (values - restored).squeeze(-2)


def _resolve_activation(name: str) -> ActivationFn:
    if name == "relu":
        return cast(ActivationFn, functional.relu)
    return cast(ActivationFn, functional.gelu)


class _ResidualAttention(nn.Module):
    def __init__(self, *, attention_dropout: float) -> None:
        super().__init__()
        self.scale: float | None = None
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, _, head_dim = queries.shape
        scale = self.scale or 1.0 / math.sqrt(head_dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attention = torch.softmax(scale * scores, dim=-1)
        dropped = self.dropout(attention)
        output = torch.einsum("bhls,bshd->blhd", dropped, values)
        return output.contiguous(), attention


class _TSMixer(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, attn_dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.attention = _ResidualAttention(attention_dropout=attn_dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_len, _ = queries.shape
        _, key_len, _ = keys.shape
        projected_queries = self.q_proj(queries).reshape(
            batch_size,
            query_len,
            self.n_heads,
            -1,
        )
        projected_keys = self.k_proj(keys).reshape(
            batch_size,
            key_len,
            self.n_heads,
            -1,
        )
        projected_values = self.v_proj(values).reshape(
            batch_size,
            key_len,
            self.n_heads,
            -1,
        )
        mixed, attention = self.attention(
            projected_queries,
            projected_keys,
            projected_values,
        )
        output = mixed.reshape(batch_size, query_len, -1)
        return self.out_proj(output), attention


class _TimeBridgeEncoder(nn.Module):
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        attentions: list[torch.Tensor | None] = []
        encoded = tokens
        for layer in self.layers:
            encoded, attention = cast(
                tuple[torch.Tensor, torch.Tensor | None],
                layer(encoded),
            )
            attentions.append(attention)
        return encoded, attentions


class _PatchEmbed(nn.Module):
    def __init__(
        self,
        *,
        input_steps: int,
        period: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_patches = input_steps // period
        self.period = period
        self.projection = nn.Sequential(
            nn.Linear(period, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        signals: torch.Tensor,
        shared_time_features: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.cat([signals, shared_time_features], dim=-1).transpose(-1, -2)
        reshaped = tokens.reshape(
            tokens.shape[0],
            tokens.shape[1],
            self.num_patches,
            self.period,
        )
        return cast(torch.Tensor, self.projection(reshaped))


class _IntegratedAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        attn_dropout: float,
        stable_len: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.stable_len = stable_len
        self.mixer = _TSMixer(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
        )
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = _resolve_activation(activation)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, None]:
        attended = self._temporal_attention(tokens)
        residual = tokens + self.dropout(attended)
        normalized = self.norm1(residual)
        ff = self.dropout(self.activation_fn(self.fc1(normalized)))
        ff = self.dropout(self.fc2(ff))
        return self.norm2(normalized + ff), None

    def _temporal_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, channel_count, patch_count, d_model = tokens.shape
        flat_tokens = tokens.reshape(batch_size * channel_count, patch_count, d_model)
        stable_tokens = _period_norm(flat_tokens, self.stable_len)
        mixed, _ = self.mixer(stable_tokens, stable_tokens, flat_tokens)
        return cast(
            torch.Tensor,
            mixed.reshape(batch_size, channel_count, patch_count, d_model),
        )


class _PatchSamplingBlock(nn.Module):
    def __init__(
        self,
        *,
        in_patches: int,
        out_patches: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        attn_dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.out_patches = out_patches
        self.mixer = _TSMixer(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
        )
        self.conv1 = nn.Conv1d(in_patches, out_patches, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_patches + 1, out_patches, kernel_size=1, bias=False)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = _resolve_activation(activation)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, None]:
        downsampled = self._downsample_attention(tokens)
        normalized = self.norm1(downsampled)
        ff = self.dropout(self.activation_fn(self.fc1(normalized)))
        ff = self.dropout(self.fc2(ff))
        return self.norm2(normalized + ff), None

    def _downsample_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, channel_count, patch_count, d_model = tokens.shape
        flat_tokens = tokens.reshape(batch_size * channel_count, patch_count, d_model)
        sampled = self.conv1(flat_tokens)
        sampled = (
            self.conv2(
                torch.cat(
                    [sampled, flat_tokens.mean(dim=-2, keepdim=True)],
                    dim=-2,
                )
            )
            + sampled
        )
        attended, _ = self.mixer(sampled, flat_tokens, flat_tokens)
        residual = self.dropout(
            sampled.reshape(
                batch_size,
                channel_count,
                self.out_patches,
                d_model,
            )
        )
        return cast(
            torch.Tensor,
            attended.reshape(
                batch_size,
                channel_count,
                self.out_patches,
                d_model,
            )
            + residual,
        )


class _CointegratedAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        total_channels: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        attn_dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.num_rc = math.ceil(total_channels**0.5)
        self.padded_channels = self.num_rc**2
        self.mixer_rows = _TSMixer(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
        )
        self.mixer_cols = _TSMixer(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
        )
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = _resolve_activation(activation)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, None]:
        attended = self._axial_attention(tokens)
        residual = tokens + self.dropout(attended)
        normalized = self.norm1(residual)
        ff = self.dropout(self.activation_fn(self.fc1(normalized)))
        ff = self.dropout(self.fc2(ff))
        return self.norm2(normalized + ff), None

    def _axial_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, channel_count, patch_count, d_model = tokens.shape
        patch_major = tokens.permute(0, 2, 1, 3).reshape(
            batch_size * patch_count,
            channel_count,
            d_model,
        )
        padded = functional.pad(
            patch_major.transpose(-1, -2),
            (0, self.padded_channels - channel_count),
        ).transpose(-1, -2)
        grid = padded.reshape(
            batch_size * patch_count,
            self.num_rc,
            self.num_rc,
            d_model,
        )
        row_tokens = grid.reshape(
            batch_size * patch_count * self.num_rc,
            self.num_rc,
            d_model,
        )
        row_attended, _ = self.mixer_rows(row_tokens, row_tokens, row_tokens)
        col_tokens = row_attended.reshape(
            batch_size * patch_count,
            self.num_rc,
            self.num_rc,
            d_model,
        ).transpose(1, 2)
        col_tokens = col_tokens.reshape(
            batch_size * patch_count * self.num_rc,
            self.num_rc,
            d_model,
        )
        col_attended, _ = self.mixer_cols(col_tokens, col_tokens, col_tokens)
        col_attended = col_attended + col_tokens
        restored = col_attended.reshape(
            batch_size * patch_count,
            self.num_rc,
            self.num_rc,
            d_model,
        ).transpose(1, 2)
        restored = restored.reshape(
            batch_size * patch_count,
            self.padded_channels,
            d_model,
        )
        restored = restored[:, :channel_count, :]
        return cast(
            torch.Tensor,
            restored.reshape(
                batch_size,
                patch_count,
                channel_count,
                d_model,
            ).permute(0, 2, 1, 3),
        )


class TimeBridgeModel(nn.Module):
    """TimeBridge for unified airport wind-speed forecasting."""

    def __init__(
        self,
        *,
        input_size: int,
        input_steps: int,
        forecast_steps: int,
        airport_count: int,
        target_size: int,
        period: int,
        num_p: int,
        ia_layers: int,
        pd_layers: int,
        ca_layers: int,
        stable_len: int,
        input_feature_count: int,
        shared_time_feature_count: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        attn_dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if input_size % airport_count != 0:
            raise ValueError("input_size must be divisible by airport_count.")
        if input_steps % period != 0:
            raise ValueError("input_steps must be divisible by period.")
        if input_size % input_feature_count != 0:
            raise ValueError("input_size must be divisible by input_feature_count.")
        if shared_time_feature_count < 0:
            raise ValueError("shared_time_feature_count must be non-negative.")
        feature_count = input_feature_count
        if shared_time_feature_count >= feature_count:
            raise ValueError(
                "shared_time_feature_count must be smaller than "
                "per-airport feature count."
            )
        input_patch_count = input_steps // period
        if num_p <= 0 or num_p > input_patch_count:
            raise ValueError("num_p must be in [1, input_steps // period].")
        if activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'.")

        self.input_size = input_size
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.airport_count = airport_count
        self.target_size = target_size
        self.period = period
        self.num_p = num_p
        self.ia_layers = ia_layers
        self.pd_layers = pd_layers
        self.ca_layers = ca_layers
        self.stable_len = stable_len
        self.input_feature_count = input_feature_count
        self.shared_time_feature_count = shared_time_feature_count
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.activation = activation
        self.feature_count = feature_count
        self.input_airport_count = input_size // input_feature_count
        self.signal_feature_count = feature_count - shared_time_feature_count
        self.signal_channel_count = self.input_airport_count * self.signal_feature_count
        self.total_channel_count = self.signal_channel_count + shared_time_feature_count
        self.input_patch_count = input_patch_count
        self.output_patch_count = input_patch_count if pd_layers == 0 else num_p

        self.embedding = _PatchEmbed(
            input_steps=input_steps,
            period=period,
            d_model=d_model,
            dropout=dropout,
        )
        self.encoder = _TimeBridgeEncoder(self._build_layers())
        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.output_patch_count * d_model, forecast_steps, bias=False),
        )
        self.output_projection = nn.Linear(
            self.signal_channel_count,
            airport_count * target_size,
        )

    @property
    def init_kwargs(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "input_steps": self.input_steps,
            "forecast_steps": self.forecast_steps,
            "airport_count": self.airport_count,
            "target_size": self.target_size,
            "period": self.period,
            "num_p": self.num_p,
            "ia_layers": self.ia_layers,
            "pd_layers": self.pd_layers,
            "ca_layers": self.ca_layers,
            "stable_len": self.stable_len,
            "input_feature_count": self.input_feature_count,
            "shared_time_feature_count": self.shared_time_feature_count,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "attn_dropout": self.attn_dropout,
            "activation": self.activation,
        }

    def forward(self, inputs: torch.Tensor) -> dict[str, Any]:
        batch_size, _ = validate_forecast_input(inputs, self.input_size)
        signals, shared_time = self._split_inputs(inputs)

        mean = signals.mean(dim=1, keepdim=True).detach()
        std = signals.std(dim=1, keepdim=True).detach()
        normalized = (signals - mean) / (std + 1e-5)

        embedded = self.embedding(normalized, shared_time)
        encoded, attentions = self.encoder(embedded)
        encoded_signals = encoded[:, : self.signal_channel_count, ...]
        decoded = self.decoder(encoded_signals).transpose(-1, -2)
        restored = decoded * std + mean
        projected = self.output_projection(restored)
        prediction = reshape_prediction(
            projected,
            batch_size=batch_size,
            forecast_steps=self.forecast_steps,
            airport_count=self.airport_count,
            target_size=self.target_size,
        )
        return {
            "prediction": prediction,
            "aux": {
                "encoded": encoded_signals,
                "attentions": attentions,
                "signal_mean": mean,
                "signal_std": std,
            },
        }

    def _split_inputs(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        signal_values = inputs[:, :, :, : self.signal_feature_count].reshape(
            inputs.shape[0],
            inputs.shape[1],
            self.signal_channel_count,
        )
        if self.shared_time_feature_count == 0:
            shared_time = signal_values.new_zeros(
                inputs.shape[0],
                inputs.shape[1],
                0,
            )
        else:
            shared_time = inputs[:, :, 0, self.signal_feature_count :]
        return signal_values, shared_time

    def _build_layers(self) -> list[nn.Module]:
        layers: list[nn.Module] = []
        for _ in range(self.ia_layers):
            layers.append(
                _IntegratedAttentionBlock(
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    n_heads=self.n_heads,
                    dropout=self.dropout,
                    attn_dropout=self.attn_dropout,
                    stable_len=self.stable_len,
                    activation=self.activation,
                )
            )
        for layer_index in range(self.pd_layers):
            in_patches = self.input_patch_count if layer_index == 0 else self.num_p
            layers.append(
                _PatchSamplingBlock(
                    in_patches=in_patches,
                    out_patches=self.num_p,
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    n_heads=self.n_heads,
                    dropout=self.dropout,
                    attn_dropout=self.attn_dropout,
                    activation=self.activation,
                )
            )
        for _ in range(self.ca_layers):
            layers.append(
                _CointegratedAttentionBlock(
                    total_channels=self.total_channel_count,
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    n_heads=self.n_heads,
                    dropout=self.dropout,
                    attn_dropout=self.attn_dropout,
                    activation=self.activation,
                )
            )
        return layers


if "timebridge" not in MODELS.keys():
    MODELS.register("timebridge", TimeBridgeModel)
