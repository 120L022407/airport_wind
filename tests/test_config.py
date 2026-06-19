from __future__ import annotations

from pathlib import Path

import pytest

from windlab.config import ConfigError, load_config


def test_load_baseline_config() -> None:
    config = load_config("config/gru/baseline_hourly.yaml")
    assert config.data.source == "series"
    assert config.model.name == "gru"
    assert config.data.target_airports == ["ZGSZ"]
    assert config.loss.name == "composite"
    assert len(config.loss.terms) == 1
    assert config.loss.terms[0].name == "mse"
    assert config.model.parameters["hidden_size"] == 64
    assert config.data.input_steps == 24
    assert config.data.forecast_steps == 24
    assert config.trainer.batch_size == 128


def test_load_gru_facl_config() -> None:
    config = load_config("config/gru/baseline_hourly_facl.yaml")
    assert config.model.name == "gru"
    assert config.loss.name == "composite"
    assert [term.name for term in config.loss.terms] == [
        "mse",
        "fourier_amplitude_correlation",
    ]
    assert config.loss.terms[1].params["mode"] == "paper_random"
    assert config.loss.terms[1].params["alpha"] == 0.1
    assert config.loss.terms[1].params["mask_mode"] == "strict_real_only"
    assert config.trainer.batch_size == 128


def test_load_gru_15min_config() -> None:
    config = load_config("config/gru/baseline_15min.yaml")
    assert config.data.source == "series_15min"
    assert config.data.time_resolution == "15min"
    assert config.data.target_airports == ["ZGSZ"]
    assert config.data.input_steps == 96
    assert config.data.forecast_steps == 96
    assert config.trainer.batch_size == 128


def test_load_gru_15min_cube_config() -> None:
    config = load_config("config/gru/baseline_15min_cube.yaml")
    assert config.data.source == "series_15min_cubic"
    assert config.data.time_resolution == "15min"
    assert config.data.target_airports == ["ZGSZ"]
    assert config.data.input_steps == 96
    assert config.data.forecast_steps == 96
    assert config.trainer.batch_size == 128


@pytest.mark.parametrize(
    "config_path",
    [
        "config/gru/baseline_15min_facl.yaml",
        "config/hcan/baseline_15min_facl.yaml",
        "config/patchtst/baseline_15min_facl.yaml",
        "config/itransformer/baseline_15min_facl.yaml",
        "config/dlinear/baseline_15min_facl.yaml",
        "config/tfps/baseline_15min_facl.yaml",
        "config/timebridge/baseline_15min_facl.yaml",
        "config/gru/baseline_15min_cube_facl.yaml",
        "config/hcan/baseline_15min_cube_facl.yaml",
        "config/patchtst/baseline_15min_cube_facl.yaml",
        "config/itransformer/baseline_15min_cube_facl.yaml",
        "config/dlinear/baseline_15min_cube_facl.yaml",
        "config/tfps/baseline_15min_cube_facl.yaml",
        "config/timebridge/baseline_15min_cube_facl.yaml",
    ],
)
def test_load_15min_facl_configs_use_all_points_mask_mode(config_path: str) -> None:
    config = load_config(config_path)
    fourier_term = next(
        term
        for term in config.loss.terms
        if term.name == "fourier_amplitude_correlation"
    )
    assert fourier_term.params["mask_mode"] == "all_points"


@pytest.mark.parametrize(
    "config_path",
    [
        "config/gru/baseline_hourly_facl.yaml",
        "config/hcan/baseline_hourly_facl.yaml",
        "config/patchtst/baseline_hourly_facl.yaml",
        "config/itransformer/baseline_hourly_facl.yaml",
        "config/dlinear/baseline_hourly_facl.yaml",
        "config/tfps/baseline_hourly_facl.yaml",
        "config/timebridge/baseline_hourly_facl.yaml",
    ],
)
def test_load_hourly_facl_configs_use_strict_real_only_mask_mode(
    config_path: str,
) -> None:
    config = load_config(config_path)
    fourier_term = next(
        term
        for term in config.loss.terms
        if term.name == "fourier_amplitude_correlation"
    )
    assert fourier_term.params["mask_mode"] == "strict_real_only"
    assert config.trainer.batch_size == 128


@pytest.mark.parametrize(
    ("config_path", "model_name", "expected_steps"),
    [
        ("config/gru/baseline_15min_facl.yaml", "gru", 96),
        ("config/gru/baseline_15min_cube_facl.yaml", "gru", 96),
        ("config/gru/baseline_hourly_facl.yaml", "gru", 24),
        ("config/gru/baseline_15min.yaml", "gru", 96),
        ("config/gru/baseline_15min_cube.yaml", "gru", 96),
        ("config/hcan/baseline_15min.yaml", "hcan", 96),
        ("config/hcan/baseline_15min_facl.yaml", "hcan", 96),
        ("config/hcan/baseline_15min_cube.yaml", "hcan", 96),
        ("config/hcan/baseline_15min_cube_facl.yaml", "hcan", 96),
        ("config/hcan/baseline_hourly_facl.yaml", "hcan", 24),
        ("config/hcan/baseline_hourly.yaml", "hcan", 24),
        ("config/hcan/no_hcl_hourly.yaml", "hcan", 24),
        ("config/patchtst/baseline_15min_facl.yaml", "patchtst", 96),
        ("config/patchtst/baseline_15min.yaml", "patchtst", 96),
        ("config/patchtst/baseline_15min_cube.yaml", "patchtst", 96),
        ("config/patchtst/baseline_15min_cube_facl.yaml", "patchtst", 96),
        ("config/patchtst/baseline_hourly_facl.yaml", "patchtst", 24),
        ("config/itransformer/baseline_15min_facl.yaml", "itransformer", 96),
        ("config/itransformer/baseline_15min.yaml", "itransformer", 96),
        ("config/itransformer/baseline_15min_cube.yaml", "itransformer", 96),
        ("config/itransformer/baseline_15min_cube_facl.yaml", "itransformer", 96),
        ("config/itransformer/baseline_hourly_facl.yaml", "itransformer", 24),
        ("config/dlinear/baseline_15min_facl.yaml", "dlinear", 96),
        ("config/dlinear/baseline_15min.yaml", "dlinear", 96),
        ("config/dlinear/baseline_15min_cube.yaml", "dlinear", 96),
        ("config/dlinear/baseline_15min_cube_facl.yaml", "dlinear", 96),
        ("config/dlinear/baseline_hourly_facl.yaml", "dlinear", 24),
        ("config/tfps/baseline_15min_facl.yaml", "tfps", 96),
        ("config/tfps/baseline_15min.yaml", "tfps", 96),
        ("config/tfps/baseline_15min_cube.yaml", "tfps", 96),
        ("config/tfps/baseline_15min_cube_facl.yaml", "tfps", 96),
        ("config/tfps/baseline_hourly_facl.yaml", "tfps", 24),
        ("config/timebridge/baseline_15min_facl.yaml", "timebridge", 96),
        ("config/timebridge/baseline_15min.yaml", "timebridge", 96),
        ("config/timebridge/baseline_15min_cube.yaml", "timebridge", 96),
        ("config/timebridge/baseline_15min_cube_facl.yaml", "timebridge", 96),
        ("config/timebridge/baseline_hourly_facl.yaml", "timebridge", 24),
        ("config/patchtst/baseline_hourly.yaml", "patchtst", 24),
        ("config/itransformer/baseline_hourly.yaml", "itransformer", 24),
        ("config/dlinear/baseline_hourly.yaml", "dlinear", 24),
        ("config/tfps/baseline_hourly.yaml", "tfps", 24),
        ("config/tfps/time_only_hourly.yaml", "tfps", 24),
        ("config/tfps/no_pattern_experts_hourly.yaml", "tfps", 24),
        ("config/timebridge/baseline_hourly.yaml", "timebridge", 24),
        ("config/timebridge/no_cointegration_hourly.yaml", "timebridge", 24),
    ],
)
def test_load_additional_model_configs(
    config_path: str,
    model_name: str,
    expected_steps: int,
) -> None:
    config = load_config(config_path)
    assert config.model.name == model_name
    assert config.data.input_steps == expected_steps
    assert config.data.forecast_steps == expected_steps
    assert config.trainer.batch_size == 128


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


def test_config_rejects_target_airport_outside_inputs(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_target_airport.yaml"
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
  target_airports: [ZGGG]
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
model:
  name: gru
  hidden_size: 8
  num_layers: 1
  dropout: 0.0
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
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="target_airports"):
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


def test_config_rejects_hcan_non_hierarchical_class_ratio(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_hcan.yaml"
    config_path.write_text(
        _valid_config_text(
            """
model:
  name: hcan
  backbone_hidden_size: 16
  backbone_num_layers: 1
  backbone_dropout: 0.0
  hidden_dim: 8
  num_coarse: 4
  num_fine: 6
  lambda_cls: 1.0
  lambda_reg: 1.0
  lambda_acl: 1.0
  lambda_direct: 1.0
"""
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="num_fine"):
        load_config(config_path)


def test_config_rejects_unknown_loss_term(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_loss_term.yaml"
    config_path.write_text(
        _valid_config_text(
            """
loss:
  name: composite
  terms:
    - name: unknown_loss
      weight: 1.0
model:
  name: gru
  hidden_size: 16
  num_layers: 1
  dropout: 0.0
"""
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="loss.terms\\[0\\].name"):
        load_config(config_path)


def test_config_rejects_facl_alpha_out_of_range(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_facl_alpha.yaml"
    config_path.write_text(
        _valid_config_text(
            """
loss:
  name: composite
  terms:
    - name: mse
      weight: 1.0
    - name: fourier_amplitude_correlation
      weight: 0.2
      params:
        mode: paper_random
        alpha: 1.5
model:
  name: gru
  hidden_size: 16
  num_layers: 1
  dropout: 0.0
"""
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="alpha"):
        load_config(config_path)


def test_config_rejects_invalid_fourier_mask_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_facl_mask_mode.yaml"
    config_path.write_text(
        _valid_config_text(
            """
loss:
  name: composite
  terms:
    - name: mse
      weight: 1.0
    - name: fourier_amplitude_correlation
      weight: 0.2
      params:
        mode: fal
        mask_mode: invalid
model:
  name: gru
  hidden_size: 16
  num_layers: 1
  dropout: 0.0
"""
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="mask_mode"):
        load_config(config_path)


def test_config_rejects_mse_with_hcan_auxiliary(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_hcan_loss_combo.yaml"
    config_path.write_text(
        _valid_config_text(
            """
loss:
  name: composite
  terms:
    - name: mse
      weight: 1.0
    - name: hcan_auxiliary
      weight: 1.0
model:
  name: gru
  hidden_size: 16
  num_layers: 1
  dropout: 0.0
"""
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="hcan_auxiliary"):
        load_config(config_path)


def test_config_rejects_tfps_expert_top_k_mismatch(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_tfps_top_k.yaml"
    config_path.write_text(
        _valid_config_text(_tfps_model_block("time_top_k: 5")),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="time_top_k"):
        load_config(config_path)


def test_config_rejects_tfps_without_active_domain(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_tfps_domains.yaml"
    config_path.write_text(
        _valid_config_text(
            _tfps_model_block("use_time_domain: false\n  use_frequency_domain: false")
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="domain"):
        load_config(config_path)


def test_config_rejects_tfps_experts_without_identifier(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_tfps_identifier.yaml"
    config_path.write_text(
        _valid_config_text(
            _tfps_model_block(
                "use_pattern_identifier: false\n  use_pattern_experts: true"
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="use_pattern_experts"):
        load_config(config_path)


def test_config_rejects_tfps_incompatible_subspace_size(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_tfps_subspace.yaml"
    config_path.write_text(
        _valid_config_text(_tfps_model_block("time_num_experts: 3")),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="time_num_experts"):
        load_config(config_path)


def test_config_rejects_timebridge_non_divisible_period(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_timebridge_period.yaml"
    config_path.write_text(
        _valid_config_text(_timebridge_model_block("period: 5")),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="period"):
        load_config(config_path)


def test_config_rejects_timebridge_num_p_too_large(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_timebridge_num_p.yaml"
    config_path.write_text(
        _valid_config_text(_timebridge_model_block("num_p: 5")),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="num_p"):
        load_config(config_path)


def test_config_rejects_timebridge_shared_time_feature_count_too_large(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "invalid_timebridge_shared_time.yaml"
    config_path.write_text(
        _valid_config_text(_timebridge_model_block("shared_time_feature_count: 1")),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="shared_time_feature_count"):
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


def _tfps_model_block(override_line: str) -> str:
    replacements = {
        line.split(":", 1)[0].strip(): line.strip()
        for line in override_line.splitlines()
        if line.strip()
    }
    lines = [
        "model:",
        "  name: tfps",
        "  d_model: 16",
        "  num_layers: 1",
        "  n_heads: 4",
        "  ff_dim: 32",
        "  dropout: 0.0",
        "  patch_len: 6",
        "  stride: 3",
        "  time_num_experts: 2",
        "  time_top_k: 1",
        "  frequency_num_experts: 2",
        "  frequency_top_k: 1",
        "  expert_hidden_size: 32",
        "  subspace_eta: 5.0",
        "  use_time_domain: true",
        "  use_frequency_domain: true",
        "  use_pattern_identifier: true",
        "  use_pattern_experts: true",
        "  noisy_gating: false",
    ]
    updated_lines = []
    for line in lines:
        key = line.split(":", 1)[0].strip()
        updated_lines.append(f"  {replacements[key]}" if key in replacements else line)
    return "\n".join(updated_lines) + "\n"


def _timebridge_model_block(override_line: str) -> str:
    replacements = {
        line.split(":", 1)[0].strip(): line.strip()
        for line in override_line.splitlines()
        if line.strip()
    }
    lines = [
        "model:",
        "  name: timebridge",
        "  period: 6",
        "  num_p: 2",
        "  ia_layers: 1",
        "  pd_layers: 1",
        "  ca_layers: 1",
        "  stable_len: 6",
        "  input_feature_count: 1",
        "  shared_time_feature_count: 0",
        "  d_model: 16",
        "  n_heads: 4",
        "  d_ff: 32",
        "  dropout: 0.0",
        "  attn_dropout: 0.1",
        "  activation: gelu",
    ]
    updated_lines = []
    for line in lines:
        key = line.split(":", 1)[0].strip()
        updated_lines.append(f"  {replacements[key]}" if key in replacements else line)
    return "\n".join(updated_lines) + "\n"
