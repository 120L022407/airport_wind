"""Experiment configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

import yaml

SUPPORTED_SOURCES = {"series", "series_15min", "EC"}
SUPPORTED_NORMALIZATION_METHODS = {"zscore"}
SUPPORTED_FIT_SPLITS = {"train"}
SUPPORTED_MODELS = {"gru", "patchtst", "itransformer", "dlinear"}

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


def _validate_data_section(raw: dict[str, Any]) -> DataSection:
    source = raw.get("source")
    if not isinstance(source, str) or source not in SUPPORTED_SOURCES:
        raise ConfigError(f"data.source must be one of {sorted(SUPPORTED_SOURCES)}.")

    root = raw.get("root")
    if not isinstance(root, str) or not root:
        raise ConfigError("data.root must be a non-empty string.")

    airports = _require_string_list(raw.get("airports"), "data.airports")
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


def _require_model_positive_int(raw: dict[str, Any], field_name: str) -> int:
    return _require_positive_int(raw.get(field_name), f"model.{field_name}")


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
    normalization_raw = _require_mapping(
        raw_yaml.get("normalization"), "normalization"
    )
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
    model_section = _validate_model_section(model_raw)
    if (
        model_section.name == "patchtst"
        and int(model_section.parameters["patch_len"]) > data_section.input_steps
    ):
        raise ConfigError("model.patch_len must be <= data.input_steps.")

    device = trainer_raw.get("device")
    if not isinstance(device, str) or not device:
        raise ConfigError("trainer.device must be a non-empty string.")
    batch_size = _require_positive_int(trainer_raw.get("batch_size"), "trainer.batch_size")
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
