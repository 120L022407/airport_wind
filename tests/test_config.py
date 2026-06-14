from __future__ import annotations

from pathlib import Path

import pytest

from windlab.config import ConfigError, load_config


def test_load_baseline_config() -> None:
    config = load_config("config/gru/baseline_hourly.yaml")
    assert config.data.source == "series"
    assert config.model.name == "gru"
    assert config.model.parameters["hidden_size"] == 64
    assert config.data.input_steps == 24
    assert config.data.forecast_steps == 24


@pytest.mark.parametrize(
    ("config_path", "model_name"),
    [
        ("config/patchtst/baseline_hourly.yaml", "patchtst"),
        ("config/itransformer/baseline_hourly.yaml", "itransformer"),
        ("config/dlinear/baseline_hourly.yaml", "dlinear"),
    ],
)
def test_load_additional_model_configs(config_path: str, model_name: str) -> None:
    config = load_config(config_path)
    assert config.model.name == model_name
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


def test_config_rejects_transformer_head_mismatch(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_heads.yaml"
    config_path.write_text(
        _valid_config_text(
            """
model:
  name: patchtst
  d_model: 10
  num_layers: 1
  n_heads: 4
  ff_dim: 16
  dropout: 0.0
  patch_len: 6
  stride: 3
"""
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="divisible"):
        load_config(config_path)


def test_config_rejects_patch_len_larger_than_input_steps(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_patch.yaml"
    config_path.write_text(
        _valid_config_text(
            """
model:
  name: patchtst
  d_model: 16
  num_layers: 1
  n_heads: 4
  ff_dim: 32
  dropout: 0.0
  patch_len: 25
  stride: 1
"""
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="patch_len"):
        load_config(config_path)


def test_config_rejects_even_dlinear_moving_average(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_dlinear.yaml"
    config_path.write_text(
        _valid_config_text(
            """
model:
  name: dlinear
  moving_avg: 4
  individual: false
"""
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="moving_avg"):
        load_config(config_path)


def test_config_rejects_unknown_model_field(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_unknown.yaml"
    config_path.write_text(
        _valid_config_text(
            """
model:
  name: itransformer
  d_model: 16
  num_layers: 1
  n_heads: 4
  ff_dim: 32
  dropout: 0.0
  patch_len: 4
"""
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="does not support"):
        load_config(config_path)


def _valid_config_text(model_block: str) -> str:
    return f"""
experiment:
  name: valid
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
  fit_split: train
  apply_to_inputs: true
{model_block}
trainer:
  device: cpu
  batch_size: 2
  epochs: 1
  patience: 1
  learning_rate: 0.001
  weight_decay: 0.0
  min_delta: 0.0
evaluation:
  metrics: [mae]
  real_observation_only: true
"""
