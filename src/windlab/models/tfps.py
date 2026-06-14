"""TFPS model adapted to the forecast contract."""

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


class _FourierEncoderLayer(nn.Module):
    """FNet-style encoder layer with FFT mixing over patch and feature axes."""

    def __init__(self, *, d_model: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm_fft = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mixed = torch.fft.fft(torch.fft.fft(tokens, dim=1), dim=2).real
        tokens = self.norm_fft(tokens + self.dropout(mixed))
        updated = self.feed_forward(tokens)
        return cast(torch.Tensor, self.norm_ff(tokens + self.dropout(updated)))


class _FrequencyEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        d_model: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _FourierEncoderLayer(d_model=d_model, ff_dim=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, variate_count, patch_count, d_model = tokens.shape
        encoded = tokens.reshape(batch_size * variate_count, patch_count, d_model)
        for layer in self.layers:
            encoded = layer(encoded)
        return encoded.reshape(batch_size, variate_count, patch_count, d_model)


class _SubspacePatternIdentifier(nn.Module):
    """Learnable subspace affinity module following TFPS EDESC routing."""

    def __init__(self, *, feature_dim: int, num_experts: int, eta: float) -> None:
        super().__init__()
        if feature_dim % num_experts != 0:
            raise ValueError("feature_dim must be divisible by num_experts.")
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.subspace_dim = feature_dim // num_experts
        self.eta = eta
        self.subspace_bases = nn.Parameter(torch.empty(feature_dim, feature_dim))
        nn.init.orthogonal_(self.subspace_bases)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, variate_count, patch_count, d_model = tokens.shape
        features = tokens.permute(0, 2, 1, 3).reshape(
            batch_size * patch_count,
            variate_count * d_model,
        )
        affinities: list[torch.Tensor] = []
        for expert_index in range(self.num_experts):
            start = expert_index * self.subspace_dim
            end = start + self.subspace_dim
            basis = self.subspace_bases[:, start:end]
            projected = features.matmul(basis)
            affinities.append(projected.pow(2).sum(dim=1, keepdim=True))

        affinity = torch.cat(affinities, dim=1)
        affinity = (affinity + self.eta * self.subspace_dim) / (
            (self.eta + 1.0) * self.subspace_dim
        )
        affinity = affinity / affinity.sum(dim=1, keepdim=True).clamp_min(1e-12)
        refined = self._refine_affinity(affinity)
        return (
            affinity.reshape(batch_size, patch_count, self.num_experts),
            refined.reshape(batch_size, patch_count, self.num_experts),
        )

    def _refine_affinity(self, affinity: torch.Tensor) -> torch.Tensor:
        cluster_frequency = affinity.sum(dim=0, keepdim=True).clamp_min(1e-12)
        weighted = affinity.pow(2) / cluster_frequency
        return weighted / weighted.sum(dim=1, keepdim=True).clamp_min(1e-12)


class _PatternExperts(nn.Module):
    """Top-k routed MLP experts over patch-level all-variate features."""

    def __init__(
        self,
        *,
        feature_dim: int,
        expert_hidden_size: int,
        num_experts: int,
        top_k: int,
        dropout: float,
        noisy_gating: bool,
    ) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k must be <= num_experts.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(feature_dim, expert_hidden_size),
                nn.ReLU(),
                nn.Linear(expert_hidden_size, feature_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        )

    def forward(self, tokens: torch.Tensor, affinity: torch.Tensor) -> torch.Tensor:
        batch_size, variate_count, patch_count, d_model = tokens.shape
        flat_tokens = tokens.permute(0, 2, 1, 3).reshape(
            batch_size,
            patch_count,
            variate_count * d_model,
        )
        gates = self._top_k_gates(affinity)
        expert_outputs = torch.stack(
            [expert(flat_tokens) for expert in self.experts],
            dim=2,
        )
        mixed = (expert_outputs * gates.unsqueeze(-1)).sum(dim=2)
        return mixed.reshape(batch_size, patch_count, variate_count, d_model).permute(
            0,
            2,
            1,
            3,
        )

    def _top_k_gates(self, affinity: torch.Tensor) -> torch.Tensor:
        logits = affinity
        if self.noisy_gating and self.training:
            logits = logits + torch.randn_like(logits) * functional.softplus(logits)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        sparse_logits = torch.full_like(logits, float("-inf"))
        sparse_logits = sparse_logits.scatter(-1, indices, top_k_logits)
        return torch.softmax(sparse_logits, dim=-1)


class _TFPSDomainBranch(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        input_patch_count: int,
        output_patch_count: int,
        patch_len: int,
        d_model: int,
        num_layers: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        num_experts: int,
        top_k: int,
        expert_hidden_size: int,
        subspace_eta: float,
        use_pattern_identifier: bool,
        use_pattern_experts: bool,
        noisy_gating: bool,
        frequency_domain: bool,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_patch_count = output_patch_count
        self.d_model = d_model
        self.frequency_domain = frequency_domain
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 1, input_patch_count, d_model)
        )
        self.embedding_dropout = nn.Dropout(dropout)
        if frequency_domain:
            self.encoder: nn.Module = _FrequencyEncoder(
                num_layers=num_layers,
                d_model=d_model,
                ff_dim=ff_dim,
                dropout=dropout,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        self.patch_projection = nn.Linear(input_patch_count, output_patch_count)

        feature_dim = input_size * d_model
        self.pattern_identifier: _SubspacePatternIdentifier | None = None
        self.pattern_experts: _PatternExperts | None = None
        if use_pattern_identifier:
            self.pattern_identifier = _SubspacePatternIdentifier(
                feature_dim=feature_dim,
                num_experts=num_experts,
                eta=subspace_eta,
            )
        if use_pattern_experts:
            self.pattern_experts = _PatternExperts(
                feature_dim=feature_dim,
                expert_hidden_size=expert_hidden_size,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                noisy_gating=noisy_gating,
            )

    def forward(
        self,
        patches: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embedded = self.embedding_dropout(
            self.patch_embedding(patches) + self.position_embedding
        )
        encoded = self._encode(embedded)
        projected = self._project_patches(encoded)
        aux: dict[str, torch.Tensor] = {}
        if self.pattern_identifier is not None:
            affinity, refined = self.pattern_identifier(projected)
            aux["affinity"] = affinity
            aux["refined_affinity"] = refined
            if self.pattern_experts is not None:
                projected = self.pattern_experts(projected, affinity)
        if self.frequency_domain:
            projected = self._inverse_frequency(projected)
        return projected, aux

    def _encode(self, embedded: torch.Tensor) -> torch.Tensor:
        if self.frequency_domain:
            return cast(torch.Tensor, self.encoder(embedded))

        batch_size, variate_count, patch_count, d_model = embedded.shape
        tokens = embedded.reshape(batch_size * variate_count, patch_count, d_model)
        encoded = self.encoder(tokens)
        return cast(
            torch.Tensor,
            encoded.reshape(batch_size, variate_count, patch_count, d_model),
        )

    def _project_patches(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.patch_projection(tokens.permute(0, 1, 3, 2))
        return cast(torch.Tensor, projected.permute(0, 1, 3, 2))

    def _inverse_frequency(self, tokens: torch.Tensor) -> torch.Tensor:
        spectrum = tokens.permute(0, 1, 3, 2)
        time_like = torch.fft.ifft(torch.fft.ifft(spectrum, dim=-2), dim=-1).real
        return cast(torch.Tensor, time_like.permute(0, 1, 3, 2))


class TFPSModel(nn.Module):
    """Time-Frequency Pattern-Specific model for fixed-horizon forecasting."""

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
        time_num_experts: int,
        time_top_k: int,
        frequency_num_experts: int,
        frequency_top_k: int,
        expert_hidden_size: int,
        subspace_eta: float,
        use_time_domain: bool,
        use_frequency_domain: bool,
        use_pattern_identifier: bool,
        use_pattern_experts: bool,
        noisy_gating: bool,
    ) -> None:
        super().__init__()
        if patch_len > input_steps:
            raise ValueError("patch_len must be <= input_steps.")
        if patch_len > forecast_steps:
            raise ValueError("patch_len must be <= forecast_steps.")
        if not use_time_domain and not use_frequency_domain:
            raise ValueError("At least one TFPS domain branch must be enabled.")
        if use_pattern_experts and not use_pattern_identifier:
            raise ValueError("use_pattern_experts requires use_pattern_identifier.")
        if time_top_k > time_num_experts:
            raise ValueError("time_top_k must be <= time_num_experts.")
        if frequency_top_k > frequency_num_experts:
            raise ValueError("frequency_top_k must be <= frequency_num_experts.")

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
        self.time_num_experts = time_num_experts
        self.time_top_k = time_top_k
        self.frequency_num_experts = frequency_num_experts
        self.frequency_top_k = frequency_top_k
        self.expert_hidden_size = expert_hidden_size
        self.subspace_eta = subspace_eta
        self.use_time_domain = use_time_domain
        self.use_frequency_domain = use_frequency_domain
        self.use_pattern_identifier = use_pattern_identifier
        self.use_pattern_experts = use_pattern_experts
        self.noisy_gating = noisy_gating
        self.input_patch_count = ((input_steps - patch_len) // stride) + 1
        self.output_patch_count = ((forecast_steps - patch_len) // stride) + 1

        self.time_branch: _TFPSDomainBranch | None = None
        self.frequency_branch: _TFPSDomainBranch | None = None
        if use_time_domain:
            self.time_branch = _TFPSDomainBranch(
                input_size=input_size,
                input_patch_count=self.input_patch_count,
                output_patch_count=self.output_patch_count,
                patch_len=patch_len,
                d_model=d_model,
                num_layers=num_layers,
                n_heads=n_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                num_experts=time_num_experts,
                top_k=time_top_k,
                expert_hidden_size=expert_hidden_size,
                subspace_eta=subspace_eta,
                use_pattern_identifier=use_pattern_identifier,
                use_pattern_experts=use_pattern_experts,
                noisy_gating=noisy_gating,
                frequency_domain=False,
            )
        if use_frequency_domain:
            self.frequency_branch = _TFPSDomainBranch(
                input_size=input_size,
                input_patch_count=self.input_patch_count,
                output_patch_count=self.output_patch_count,
                patch_len=patch_len,
                d_model=d_model,
                num_layers=num_layers,
                n_heads=n_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                num_experts=frequency_num_experts,
                top_k=frequency_top_k,
                expert_hidden_size=expert_hidden_size,
                subspace_eta=subspace_eta,
                use_pattern_identifier=use_pattern_identifier,
                use_pattern_experts=use_pattern_experts,
                noisy_gating=noisy_gating,
                frequency_domain=True,
            )

        branch_count = int(use_time_domain) + int(use_frequency_domain)
        head_features = self.output_patch_count * d_model * branch_count
        self.horizon_head = nn.Linear(head_features, forecast_steps)
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
            "time_num_experts": self.time_num_experts,
            "time_top_k": self.time_top_k,
            "frequency_num_experts": self.frequency_num_experts,
            "frequency_top_k": self.frequency_top_k,
            "expert_hidden_size": self.expert_hidden_size,
            "subspace_eta": self.subspace_eta,
            "use_time_domain": self.use_time_domain,
            "use_frequency_domain": self.use_frequency_domain,
            "use_pattern_identifier": self.use_pattern_identifier,
            "use_pattern_experts": self.use_pattern_experts,
            "noisy_gating": self.noisy_gating,
        }

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor | dict[str, Any]]:
        batch_size, _ = validate_forecast_input(inputs, self.input_size)
        flattened = flatten_airport_features(inputs).transpose(1, 2)
        patches = flattened.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        branch_outputs: list[torch.Tensor] = []
        aux: dict[str, Any] = {}
        if self.time_branch is not None:
            time_output, time_aux = self.time_branch(patches)
            branch_outputs.append(time_output)
            aux["time"] = time_aux
        if self.frequency_branch is not None:
            frequency_output, frequency_aux = self.frequency_branch(patches)
            branch_outputs.append(frequency_output)
            aux["frequency"] = frequency_aux

        features = torch.cat(branch_outputs, dim=-1)
        per_variate_features = features.reshape(batch_size, self.input_size, -1)
        per_variate_forecast = self.horizon_head(per_variate_features).transpose(1, 2)
        projected = self.output_projection(per_variate_forecast)
        prediction = reshape_prediction(
            projected,
            batch_size=batch_size,
            forecast_steps=self.forecast_steps,
            airport_count=self.airport_count,
            target_size=self.target_size,
        )
        return {"prediction": prediction, "aux": aux}


if "tfps" not in MODELS.keys():
    MODELS.register("tfps", TFPSModel)
