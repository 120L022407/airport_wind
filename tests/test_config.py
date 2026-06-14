from __future__ import annotations

from pathlib import Path

import pytest

from windlab.config import ConfigError, load_config


def test_load_baseline_config() -> None:
    config = load_config("config/gru/baseline_hourly.yaml")
    assert config.data.source == "series"
    assert config.model.name == "gru"
    assert config.data.input_steps == 24
    assert config.data.forecast_steps == 24


def test_config_rejects_non_train_normalization(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
experiment:
  name: invalid
  seed: 1
runtime:
  output_root: outputs
data:
  root: data
  source: series
  airports: [ZGSZ]
  input_variables: [sknt]
  target_variables: [sknt]
  time_resolution: 1h
  input_steps: 24
  forecast_steps: 24
normalization:
  enabled: true
  method: zscore
  fit_split: val
  apply_to_inputs: true
model:
  name: gru
  hidden_size: 8
trainer:
  ridge_lambda: 0.001
evaluation:
  metrics: [mae]
  real_observation_only: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="fit_split"):
        load_config(config_path)


def test_config_rejects_target_not_in_inputs(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
experiment:
  name: invalid
  seed: 1
runtime:
  output_root: outputs
data:
  root: data
  source: series
  airports: [ZGSZ]
  input_variables: [gust]
  target_variables: [sknt]
  time_resolution: 1h
  input_steps: 24
  forecast_steps: 24
normalization:
  enabled: true
  method: zscore
  fit_split: train
  apply_to_inputs: true
model:
  name: gru
  hidden_size: 8
trainer:
  ridge_lambda: 0.001
evaluation:
  metrics: [mae]
  real_observation_only: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="target_variables"):
        load_config(config_path)
