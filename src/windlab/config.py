"""Experiment configuration loading and validation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_SOURCES = {"series", "series_15min", "series_15min_cubic", "EC"}
SUPPORTED_NORMALIZATION_METHODS = {"zscore"}
SUPPORTED_FIT_SPLITS = {"train"}
SUPPORTED_LOSSES = {"composite"}
SUPPORTED_LOSS_TERMS = {
    "mse",
    "hcan_auxiliary",
    "fourier_amplitude_correlation",
    "patch_wise_structural",
}
SUPPORTED_FOURIER_LOSS_MODES = {"paper_random", "fal", "fcl"}
SUPPORTED_FOURIER_MASK_MODES = {"strict_real_only", "all_points"}
SUPPORTED_PATCH_WISE_STRUCTURAL_MASK_MODES = {"strict_real_only", "all_points"}
SUPPORTED_MODELS = {
    "gru",
    "hcan",
    "patchtst",
    "itransformer",
    "dlinear",
    "tfps",
    "timebridge",
}

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.+?))?\}")


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


@dataclass(frozen=True)
class ExperimentSection:
    name: str
    seed: int


@dataclass(frozen=True)
class RuntimeSection:
    output_root: str


@dataclass(frozen=True)
class DataSection:
    root: str
    source: str
    airports: list[str]
    target_airports: list[str]
    input_variables: list[str]
    target_variables: list[str]
    time_resolution: str
    input_steps: int
    forecast_steps: int


@dataclass(frozen=True)
class NormalizationSection:
    enabled: bool
    method: str
    fit_split: str
    apply_to_inputs: bool


@dataclass(frozen=True)
class ModelSection:
    name: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class LossTermSection:
    name: str
    weight: float
    params: dict[str, Any]


@dataclass(frozen=True)
class LossSection:
    name: str
    terms: list[LossTermSection]


@dataclass(frozen=True)
class TrainerSection:
    device: str
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    min_delta: float


@dataclass(frozen=True)
class EvaluationSection:
    metrics: list[str]
    real_observation_only: bool


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentSection
    runtime: RuntimeSection
    data: DataSection
    normalization: NormalizationSection
    loss: LossSection
    model: ModelSection
    trainer: TrainerSection
    evaluation: EvaluationSection

    @property
    def run_name(self) -> str:
        return self.experiment.name


def _expand_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        env_name = match.group(1)
        default_value = match.group(2)
        env_value = os.environ.get(env_name)
        if env_value is not None:
            return env_value
        if default_value is not None:
            return default_value
        raise ConfigError(f"Environment variable {env_name} is not set.")

    return _ENV_PATTERN.sub(replace, value)


def _expand_env_values(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, list):
        return [_expand_env_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_values(item) for key, item in value.items()}
    return value


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field_name} must be a mapping.")
    return value


def _require_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{field_name} must be a list of strings.")
    if not value:
        raise ConfigError(f"{field_name} must not be empty.")
    if len(set(value)) != len(value):
        raise ConfigError(f"{field_name} must not contain duplicates.")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field_name} must be a positive integer.")
    return value


def _require_non_negative_float(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ConfigError(f"{field_name} must be a non-negative float.")
    return float(value)


def _validate_data_section(raw: dict[str, Any]) -> DataSection:
    source = raw.get("source")
    if not isinstance(source, str) or source not in SUPPORTED_SOURCES:
        raise ConfigError(f"data.source must be one of {sorted(SUPPORTED_SOURCES)}.")

    root = raw.get("root")
    if not isinstance(root, str) or not root:
        raise ConfigError("data.root must be a non-empty string.")

    airports = _require_string_list(raw.get("airports"), "data.airports")
    target_airports_raw = raw.get("target_airports")
    if target_airports_raw is None:
        target_airports = list(airports)
    else:
        target_airports = _require_string_list(
            target_airports_raw,
            "data.target_airports",
        )
    missing_target_airports = [
        airport for airport in target_airports if airport not in airports
    ]
    if missing_target_airports:
        raise ConfigError(
            "data.target_airports must be a subset of data.airports: "
            + ", ".join(missing_target_airports)
        )
    input_variables = _require_string_list(
        raw.get("input_variables"), "data.input_variables"
    )
    target_variables = _require_string_list(
        raw.get("target_variables"), "data.target_variables"
    )
    missing_targets = [name for name in target_variables if name not in input_variables]
    if missing_targets:
        raise ConfigError(
            "data.target_variables must be present in data.input_variables: "
            + ", ".join(missing_targets)
        )

    time_resolution = raw.get("time_resolution")
    if not isinstance(time_resolution, str) or not time_resolution:
        raise ConfigError("data.time_resolution must be a non-empty string.")

    return DataSection(
        root=root,
        source=source,
        airports=airports,
        target_airports=target_airports,
        input_variables=input_variables,
        target_variables=target_variables,
        time_resolution=time_resolution,
        input_steps=_require_positive_int(raw.get("input_steps"), "data.input_steps"),
        forecast_steps=_require_positive_int(
            raw.get("forecast_steps"), "data.forecast_steps"
        ),
    )


def _validate_normalization_section(raw: dict[str, Any]) -> NormalizationSection:
    enabled = raw.get("enabled")
    if not isinstance(enabled, bool):
        raise ConfigError("normalization.enabled must be a boolean.")

    method = raw.get("method")
    if not isinstance(method, str) or method not in SUPPORTED_NORMALIZATION_METHODS:
        raise ConfigError(
            "normalization.method must be one of "
            f"{sorted(SUPPORTED_NORMALIZATION_METHODS)}."
        )

    fit_split = raw.get("fit_split")
    if not isinstance(fit_split, str) or fit_split not in SUPPORTED_FIT_SPLITS:
        raise ConfigError("normalization.fit_split must be 'train'.")

    apply_to_inputs = raw.get("apply_to_inputs")
    if not isinstance(apply_to_inputs, bool):
        raise ConfigError("normalization.apply_to_inputs must be a boolean.")

    return NormalizationSection(
        enabled=enabled,
        method=method,
        fit_split=fit_split,
        apply_to_inputs=apply_to_inputs,
    )


def _validate_fourier_loss_params(raw: dict[str, Any]) -> dict[str, Any]:
    unknown_fields = sorted(set(raw) - {"mode", "alpha", "mask_mode"})
    if unknown_fields:
        raise ConfigError(
            "loss term 'fourier_amplitude_correlation' does not support fields: "
            + ", ".join(unknown_fields)
        )
    mode = raw.get("mode")
    if not isinstance(mode, str) or mode not in SUPPORTED_FOURIER_LOSS_MODES:
        raise ConfigError(
            "loss.term.params.mode must be one of ['fal', 'fcl', 'paper_random']."
        )
    mask_mode = raw.get("mask_mode", "strict_real_only")
    if (
        not isinstance(mask_mode, str)
        or mask_mode not in SUPPORTED_FOURIER_MASK_MODES
    ):
        raise ConfigError(
            "loss.term.params.mask_mode must be one of "
            "['all_points', 'strict_real_only']."
        )
    if mode == "paper_random":
        alpha = raw.get("alpha")
        if not isinstance(alpha, (int, float)) or not 0.0 <= float(alpha) <= 1.0:
            raise ConfigError("loss.term.params.alpha must be in [0.0, 1.0].")
        return {"mode": mode, "alpha": float(alpha), "mask_mode": mask_mode}
    if "alpha" in raw:
        raise ConfigError(
            "loss.term.params.alpha is only valid when mode='paper_random'."
        )
    return {"mode": mode, "mask_mode": mask_mode}


def _validate_patch_wise_structural_loss_params(raw: dict[str, Any]) -> dict[str, Any]:
    unknown_fields = sorted(set(raw) - {"patch_len_threshold", "mask_mode"})
    if unknown_fields:
        raise ConfigError(
            "loss term 'patch_wise_structural' does not support fields: "
            + ", ".join(unknown_fields)
        )
    patch_len_threshold = _require_positive_int(
        raw.get("patch_len_threshold"),
        "loss.term.params.patch_len_threshold",
    )
    mask_mode = raw.get("mask_mode")
    if (
        not isinstance(mask_mode, str)
        or mask_mode not in SUPPORTED_PATCH_WISE_STRUCTURAL_MASK_MODES
    ):
        raise ConfigError(
            "loss.term.params.mask_mode must be one of "
            "['all_points', 'strict_real_only']."
        )
    return {
        "patch_len_threshold": patch_len_threshold,
        "mask_mode": mask_mode,
    }


def _validate_loss_term(raw: Any, index: int) -> LossTermSection:
    field_name = f"loss.terms[{index}]"
    mapping = _require_mapping(raw, field_name)
    unknown_fields = sorted(set(mapping) - {"name", "weight", "params"})
    if unknown_fields:
        raise ConfigError(
            f"{field_name} does not support fields: " + ", ".join(unknown_fields)
        )
    name = mapping.get("name")
    if not isinstance(name, str) or name not in SUPPORTED_LOSS_TERMS:
        raise ConfigError(
            f"{field_name}.name must be one of {sorted(SUPPORTED_LOSS_TERMS)}."
        )
    weight = _require_non_negative_float(mapping.get("weight"), f"{field_name}.weight")
    params_raw = mapping.get("params", {})
    params = _require_mapping(params_raw, f"{field_name}.params")
    if name in {"mse", "hcan_auxiliary"}:
        if params:
            raise ConfigError(f"{field_name}.params must be empty for loss {name!r}.")
        return LossTermSection(name=name, weight=weight, params={})
    if name == "fourier_amplitude_correlation":
        validated_params = _validate_fourier_loss_params(params)
    elif name == "patch_wise_structural":
        validated_params = _validate_patch_wise_structural_loss_params(params)
    else:
        raise AssertionError(f"Unhandled loss term validator for {name!r}.")
    return LossTermSection(
        name=name,
        weight=weight,
        params=validated_params,
    )


def _default_loss_section() -> LossSection:
    return LossSection(
        name="composite",
        terms=[LossTermSection(name="mse", weight=1.0, params={})],
    )


def _validate_loss_section(raw: Any) -> LossSection:
    if raw is None:
        return _default_loss_section()
    mapping = _require_mapping(raw, "loss")
    unknown_fields = sorted(set(mapping) - {"name", "terms"})
    if unknown_fields:
        raise ConfigError("loss does not support fields: " + ", ".join(unknown_fields))
    name = mapping.get("name")
    if not isinstance(name, str) or name not in SUPPORTED_LOSSES:
        raise ConfigError(f"loss.name must be one of {sorted(SUPPORTED_LOSSES)}.")
    terms_raw = mapping.get("terms")
    if not isinstance(terms_raw, list) or not terms_raw:
        raise ConfigError("loss.terms must be a non-empty list.")
    terms = [
        _validate_loss_term(term_raw, index) for index, term_raw in enumerate(terms_raw)
    ]
    term_names = {term.name for term in terms if term.weight > 0.0}
    if "mse" in term_names and "hcan_auxiliary" in term_names:
        raise ConfigError(
            "loss.terms must not combine 'mse' with 'hcan_auxiliary' because "
            "hcan_auxiliary already includes the direct forecast term."
        )
    if not any(term.weight > 0.0 for term in terms):
        raise ConfigError("loss.terms must contain at least one positive-weight term.")
    return LossSection(name=name, terms=terms)


def _require_model_positive_int(raw: dict[str, Any], field_name: str) -> int:
    return _require_positive_int(raw.get(field_name), f"model.{field_name}")


def _require_model_non_negative_int(raw: dict[str, Any], field_name: str) -> int:
    value = raw.get(field_name)
    if not isinstance(value, int) or value < 0:
        raise ConfigError(f"model.{field_name} must be a non-negative integer.")
    return value


def _require_model_float_range(
    raw: dict[str, Any],
    field_name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = raw.get(field_name)
    if not isinstance(value, (int, float)) or not minimum <= float(value) < maximum:
        raise ConfigError(f"model.{field_name} must be in [{minimum}, {maximum}).")
    return float(value)


def _require_model_positive_float(raw: dict[str, Any], field_name: str) -> float:
    value = raw.get(field_name)
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise ConfigError(f"model.{field_name} must be a positive float.")
    return float(value)


def _require_model_non_negative_float(raw: dict[str, Any], field_name: str) -> float:
    value = raw.get(field_name)
    if not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ConfigError(f"model.{field_name} must be a non-negative float.")
    return float(value)


def _require_model_bool(raw: dict[str, Any], field_name: str) -> bool:
    value = raw.get(field_name)
    if not isinstance(value, bool):
        raise ConfigError(f"model.{field_name} must be a boolean.")
    return value


def _reject_unknown_model_fields(
    raw: dict[str, Any],
    allowed_fields: set[str],
    model_name: str,
) -> None:
    unknown_fields = sorted(set(raw) - allowed_fields)
    if unknown_fields:
        raise ConfigError(
            f"model {model_name!r} does not support fields: "
            + ", ".join(unknown_fields)
        )


def _validate_transformer_heads(d_model: int, n_heads: int) -> None:
    if d_model % n_heads != 0:
        raise ConfigError("model.d_model must be divisible by model.n_heads.")


def _validate_tfps_section(
    raw: dict[str, Any],
    transformer_fields: set[str],
) -> ModelSection:
    _reject_unknown_model_fields(
        raw,
        transformer_fields
        | {
            "patch_len",
            "stride",
            "time_num_experts",
            "time_top_k",
            "frequency_num_experts",
            "frequency_top_k",
            "expert_hidden_size",
            "subspace_eta",
            "use_time_domain",
            "use_frequency_domain",
            "use_pattern_identifier",
            "use_pattern_experts",
            "noisy_gating",
        },
        "tfps",
    )
    d_model = _require_model_positive_int(raw, "d_model")
    n_heads = _require_model_positive_int(raw, "n_heads")
    _validate_transformer_heads(d_model, n_heads)
    time_num_experts = _require_model_positive_int(raw, "time_num_experts")
    time_top_k = _require_model_positive_int(raw, "time_top_k")
    frequency_num_experts = _require_model_positive_int(
        raw,
        "frequency_num_experts",
    )
    frequency_top_k = _require_model_positive_int(raw, "frequency_top_k")
    if time_top_k > time_num_experts:
        raise ConfigError("model.time_top_k must be <= model.time_num_experts.")
    if frequency_top_k > frequency_num_experts:
        raise ConfigError(
            "model.frequency_top_k must be <= model.frequency_num_experts."
        )
    use_time_domain = _require_model_bool(raw, "use_time_domain")
    use_frequency_domain = _require_model_bool(raw, "use_frequency_domain")
    use_pattern_identifier = _require_model_bool(raw, "use_pattern_identifier")
    use_pattern_experts = _require_model_bool(raw, "use_pattern_experts")
    if not use_time_domain and not use_frequency_domain:
        raise ConfigError("At least one TFPS domain branch must be enabled.")
    if use_pattern_experts and not use_pattern_identifier:
        raise ConfigError("model.use_pattern_experts requires use_pattern_identifier.")

    return ModelSection(
        name="tfps",
        parameters={
            "d_model": d_model,
            "num_layers": _require_model_positive_int(raw, "num_layers"),
            "n_heads": n_heads,
            "ff_dim": _require_model_positive_int(raw, "ff_dim"),
            "dropout": _require_model_float_range(
                raw,
                "dropout",
                minimum=0.0,
                maximum=1.0,
            ),
            "patch_len": _require_model_positive_int(raw, "patch_len"),
            "stride": _require_model_positive_int(raw, "stride"),
            "time_num_experts": time_num_experts,
            "time_top_k": time_top_k,
            "frequency_num_experts": frequency_num_experts,
            "frequency_top_k": frequency_top_k,
            "expert_hidden_size": _require_model_positive_int(
                raw,
                "expert_hidden_size",
            ),
            "subspace_eta": _require_model_positive_float(raw, "subspace_eta"),
            "use_time_domain": use_time_domain,
            "use_frequency_domain": use_frequency_domain,
            "use_pattern_identifier": use_pattern_identifier,
            "use_pattern_experts": use_pattern_experts,
            "noisy_gating": _require_model_bool(raw, "noisy_gating"),
        },
    )


def _validate_timebridge_section(
    raw: dict[str, Any],
    transformer_fields: set[str],
) -> ModelSection:
    _ = transformer_fields
    _reject_unknown_model_fields(
        raw,
        {
            "name",
            "period",
            "num_p",
            "ia_layers",
            "pd_layers",
            "ca_layers",
            "stable_len",
            "input_feature_count",
            "d_model",
            "n_heads",
            "d_ff",
            "dropout",
            "attn_dropout",
            "shared_time_feature_count",
            "activation",
        },
        "timebridge",
    )
    d_model = _require_model_positive_int(raw, "d_model")
    n_heads = _require_model_positive_int(raw, "n_heads")
    _validate_transformer_heads(d_model, n_heads)
    activation = raw.get("activation")
    if not isinstance(activation, str) or activation not in {"relu", "gelu"}:
        raise ConfigError("model.activation must be 'relu' or 'gelu'.")

    return ModelSection(
        name="timebridge",
        parameters={
            "period": _require_model_positive_int(raw, "period"),
            "num_p": _require_model_positive_int(raw, "num_p"),
            "ia_layers": _require_model_non_negative_int(raw, "ia_layers"),
            "pd_layers": _require_model_non_negative_int(raw, "pd_layers"),
            "ca_layers": _require_model_non_negative_int(raw, "ca_layers"),
            "stable_len": _require_model_positive_int(raw, "stable_len"),
            "input_feature_count": _require_model_positive_int(
                raw,
                "input_feature_count",
            ),
            "shared_time_feature_count": _require_model_non_negative_int(
                raw,
                "shared_time_feature_count",
            ),
            "d_model": d_model,
            "n_heads": n_heads,
            "d_ff": _require_model_positive_int(raw, "d_ff"),
            "dropout": _require_model_float_range(
                raw,
                "dropout",
                minimum=0.0,
                maximum=1.0,
            ),
            "attn_dropout": _require_model_float_range(
                raw,
                "attn_dropout",
                minimum=0.0,
                maximum=1.0,
            ),
            "activation": activation,
        },
    )


def _validate_hcan_section(raw: dict[str, Any]) -> ModelSection:
    _reject_unknown_model_fields(
        raw,
        {
            "name",
            "backbone_hidden_size",
            "backbone_num_layers",
            "backbone_dropout",
            "hidden_dim",
            "num_coarse",
            "num_fine",
            "lambda_cls",
            "lambda_reg",
            "lambda_acl",
            "lambda_direct",
        },
        "hcan",
    )
    num_coarse = _require_model_positive_int(raw, "num_coarse")
    num_fine = _require_model_positive_int(raw, "num_fine")
    if num_fine != 2 * num_coarse:
        raise ConfigError("model.num_fine must equal 2 * model.num_coarse.")
    return ModelSection(
        name="hcan",
        parameters={
            "backbone_hidden_size": _require_model_positive_int(
                raw,
                "backbone_hidden_size",
            ),
            "backbone_num_layers": _require_model_positive_int(
                raw,
                "backbone_num_layers",
            ),
            "backbone_dropout": _require_model_float_range(
                raw,
                "backbone_dropout",
                minimum=0.0,
                maximum=1.0,
            ),
            "hidden_dim": _require_model_positive_int(raw, "hidden_dim"),
            "num_coarse": num_coarse,
            "num_fine": num_fine,
            "lambda_cls": _require_model_non_negative_float(raw, "lambda_cls"),
            "lambda_reg": _require_model_non_negative_float(raw, "lambda_reg"),
            "lambda_acl": _require_model_non_negative_float(raw, "lambda_acl"),
            "lambda_direct": _require_model_non_negative_float(
                raw,
                "lambda_direct",
            ),
        },
    )


def _validate_model_section(raw: dict[str, Any]) -> ModelSection:
    model_name = raw.get("name")
    if not isinstance(model_name, str) or not model_name:
        raise ConfigError("model.name must be a non-empty string.")
    if model_name not in SUPPORTED_MODELS:
        raise ConfigError(f"model.name must be one of {sorted(SUPPORTED_MODELS)}.")

    transformer_fields = {
        "name",
        "d_model",
        "num_layers",
        "n_heads",
        "ff_dim",
        "dropout",
    }
    if model_name == "gru":
        _reject_unknown_model_fields(
            raw,
            {"name", "hidden_size", "num_layers", "dropout"},
            model_name,
        )
        return ModelSection(
            name=model_name,
            parameters={
                "hidden_size": _require_model_positive_int(raw, "hidden_size"),
                "num_layers": _require_model_positive_int(raw, "num_layers"),
                "dropout": _require_model_float_range(
                    raw,
                    "dropout",
                    minimum=0.0,
                    maximum=1.0,
                ),
            },
        )
    if model_name == "hcan":
        return _validate_hcan_section(raw)
    if model_name == "patchtst":
        _reject_unknown_model_fields(
            raw,
            transformer_fields | {"patch_len", "stride"},
            model_name,
        )
        d_model = _require_model_positive_int(raw, "d_model")
        n_heads = _require_model_positive_int(raw, "n_heads")
        _validate_transformer_heads(d_model, n_heads)
        return ModelSection(
            name=model_name,
            parameters={
                "d_model": d_model,
                "num_layers": _require_model_positive_int(raw, "num_layers"),
                "n_heads": n_heads,
                "ff_dim": _require_model_positive_int(raw, "ff_dim"),
                "dropout": _require_model_float_range(
                    raw,
                    "dropout",
                    minimum=0.0,
                    maximum=1.0,
                ),
                "patch_len": _require_model_positive_int(raw, "patch_len"),
                "stride": _require_model_positive_int(raw, "stride"),
            },
        )
    if model_name == "itransformer":
        _reject_unknown_model_fields(raw, transformer_fields, model_name)
        d_model = _require_model_positive_int(raw, "d_model")
        n_heads = _require_model_positive_int(raw, "n_heads")
        _validate_transformer_heads(d_model, n_heads)
        return ModelSection(
            name=model_name,
            parameters={
                "d_model": d_model,
                "num_layers": _require_model_positive_int(raw, "num_layers"),
                "n_heads": n_heads,
                "ff_dim": _require_model_positive_int(raw, "ff_dim"),
                "dropout": _require_model_float_range(
                    raw,
                    "dropout",
                    minimum=0.0,
                    maximum=1.0,
                ),
            },
        )
    if model_name == "tfps":
        return _validate_tfps_section(raw, transformer_fields)
    if model_name == "timebridge":
        return _validate_timebridge_section(raw, transformer_fields)

    _reject_unknown_model_fields(raw, {"name", "moving_avg", "individual"}, model_name)
    individual = raw.get("individual")
    if not isinstance(individual, bool):
        raise ConfigError("model.individual must be a boolean.")
    moving_avg = _require_model_positive_int(raw, "moving_avg")
    if moving_avg % 2 == 0:
        raise ConfigError("model.moving_avg must be a positive odd integer.")
    return ModelSection(
        name=model_name,
        parameters={
            "moving_avg": moving_avg,
            "individual": individual,
        },
    )


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load and validate one experiment config."""

    raw_text = Path(config_path).read_text(encoding="utf-8")
    raw_yaml = yaml.safe_load(raw_text)
    if not isinstance(raw_yaml, dict):
        raise ConfigError("Config root must be a mapping.")
    raw_yaml = _expand_env_values(raw_yaml)

    experiment_raw = _require_mapping(raw_yaml.get("experiment"), "experiment")
    runtime_raw = _require_mapping(raw_yaml.get("runtime"), "runtime")
    data_raw = _require_mapping(raw_yaml.get("data"), "data")
    normalization_raw = _require_mapping(raw_yaml.get("normalization"), "normalization")
    loss_raw = raw_yaml.get("loss")
    model_raw = _require_mapping(raw_yaml.get("model"), "model")
    trainer_raw = _require_mapping(raw_yaml.get("trainer"), "trainer")
    evaluation_raw = _require_mapping(raw_yaml.get("evaluation"), "evaluation")

    name = experiment_raw.get("name")
    seed = experiment_raw.get("seed")
    if not isinstance(name, str) or not name:
        raise ConfigError("experiment.name must be a non-empty string.")
    if not isinstance(seed, int):
        raise ConfigError("experiment.seed must be an integer.")

    output_root = runtime_raw.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise ConfigError("runtime.output_root must be a non-empty string.")

    data_section = _validate_data_section(data_raw)
    normalization_section = _validate_normalization_section(normalization_raw)
    loss_section = _validate_loss_section(loss_raw)
    model_section = _validate_model_section(model_raw)
    if (
        model_section.name == "patchtst"
        and int(model_section.parameters["patch_len"]) > data_section.input_steps
    ):
        raise ConfigError("model.patch_len must be <= data.input_steps.")
    if model_section.name == "tfps":
        patch_len = int(model_section.parameters["patch_len"])
        if patch_len > data_section.input_steps:
            raise ConfigError("model.patch_len must be <= data.input_steps.")
        if patch_len > data_section.forecast_steps:
            raise ConfigError("model.patch_len must be <= data.forecast_steps.")
        if bool(model_section.parameters["use_pattern_identifier"]):
            flattened_input_size = len(data_section.airports) * len(
                data_section.input_variables
            )
            feature_dim = flattened_input_size * int(
                model_section.parameters["d_model"]
            )
            if bool(model_section.parameters["use_time_domain"]):
                time_num_experts = int(model_section.parameters["time_num_experts"])
                if feature_dim % time_num_experts != 0:
                    raise ConfigError(
                        "airport/input variable/d_model product must be divisible "
                        "by model.time_num_experts."
                    )
            if bool(model_section.parameters["use_frequency_domain"]):
                frequency_num_experts = int(
                    model_section.parameters["frequency_num_experts"]
                )
                if feature_dim % frequency_num_experts != 0:
                    raise ConfigError(
                        "airport/input variable/d_model product must be divisible "
                        "by model.frequency_num_experts."
                    )
    if model_section.name == "timebridge":
        period = int(model_section.parameters["period"])
        if data_section.input_steps % period != 0:
            raise ConfigError("model.period must divide data.input_steps.")
        input_patch_count = data_section.input_steps // period
        if int(model_section.parameters["num_p"]) > input_patch_count:
            raise ConfigError(
                "model.num_p must be <= data.input_steps // model.period."
            )
        shared_time_feature_count = int(
            model_section.parameters["shared_time_feature_count"]
        )
        input_feature_count = int(model_section.parameters["input_feature_count"])
        if input_feature_count != len(data_section.input_variables):
            raise ConfigError(
                "model.input_feature_count must match data.input_variables length."
            )
        if shared_time_feature_count >= input_feature_count:
            raise ConfigError(
                "model.shared_time_feature_count must be smaller than "
                "model.input_feature_count."
            )

    device = trainer_raw.get("device")
    if not isinstance(device, str) or not device:
        raise ConfigError("trainer.device must be a non-empty string.")
    batch_size = _require_positive_int(
        trainer_raw.get("batch_size"),
        "trainer.batch_size",
    )
    epochs = _require_positive_int(trainer_raw.get("epochs"), "trainer.epochs")
    patience = _require_positive_int(trainer_raw.get("patience"), "trainer.patience")
    learning_rate = trainer_raw.get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or float(learning_rate) <= 0.0:
        raise ConfigError("trainer.learning_rate must be a positive float.")
    weight_decay = trainer_raw.get("weight_decay")
    if not isinstance(weight_decay, (int, float)) or float(weight_decay) < 0.0:
        raise ConfigError("trainer.weight_decay must be a non-negative float.")
    min_delta = trainer_raw.get("min_delta")
    if not isinstance(min_delta, (int, float)) or float(min_delta) < 0.0:
        raise ConfigError("trainer.min_delta must be a non-negative float.")

    metrics = _require_string_list(evaluation_raw.get("metrics"), "evaluation.metrics")
    real_observation_only = evaluation_raw.get("real_observation_only")
    if not isinstance(real_observation_only, bool):
        raise ConfigError("evaluation.real_observation_only must be a boolean.")

    return ExperimentConfig(
        experiment=ExperimentSection(name=name, seed=seed),
        runtime=RuntimeSection(output_root=output_root),
        data=data_section,
        normalization=normalization_section,
        loss=loss_section,
        model=model_section,
        trainer=TrainerSection(
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            min_delta=float(min_delta),
        ),
        evaluation=EvaluationSection(
            metrics=metrics,
            real_observation_only=real_observation_only,
        ),
    )
