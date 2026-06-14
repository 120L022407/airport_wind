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
    hidden_size: int
    num_layers: int
    dropout: float


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

    model_name = model_raw.get("name")
    hidden_size = model_raw.get("hidden_size")
    num_layers = model_raw.get("num_layers")
    dropout = model_raw.get("dropout")
    if not isinstance(model_name, str) or not model_name:
        raise ConfigError("model.name must be a non-empty string.")
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ConfigError("model.hidden_size must be a positive integer.")
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ConfigError("model.num_layers must be a positive integer.")
    if not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
        raise ConfigError("model.dropout must be in [0.0, 1.0).")

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
        data=_validate_data_section(data_raw),
        normalization=_validate_normalization_section(normalization_raw),
        model=ModelSection(
            name=model_name,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=float(dropout),
        ),
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
