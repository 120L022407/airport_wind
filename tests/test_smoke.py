from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .helpers import create_series_fixture, create_synthetic_data_root


def test_training_and_evaluation_smoke(tmp_path: Path) -> None:
    data_root = create_synthetic_data_root(tmp_path)
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        f"""
experiment:
  name: smoke_gru
  seed: 11
runtime:
  output_root: {output_root}
data:
  root: {data_root}
  source: series
  airports: [ZGSZ, ZGGG, VHHH, VMMC]
  target_airports: [ZGSZ]
  input_variables:
    [
      sknt,
      wind_dir_sin,
      wind_dir_cos,
      tmpf,
      alti,
      dwpf,
      relh,
      gust,
      hour_sin,
      hour_cos,
      month_sin,
      month_cos,
      day_of_year_norm,
    ]
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
  batch_size: 4
  epochs: 2
  patience: 2
  learning_rate: 0.001
  weight_decay: 0.0
  min_delta: 0.0
evaluation:
  metrics: [mae, rmse, bias]
  real_observation_only: true
""",
        encoding="utf-8",
    )
    train_result = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Train epochs" in train_result.stderr
    assert "Predict test" in train_result.stderr
    run_dir = Path(train_result.stdout.strip())
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "checkpoint.pt").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "normalization.npz").is_file()

    eval_result = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--run-dir", str(run_dir)],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Predict val" in eval_result.stderr
    assert "Predict test" in eval_result.stderr
    metrics = json.loads(eval_result.stdout)
    assert set(metrics["validation"]) == {"mae", "rmse", "bias"}

    saved_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics["real_observation_only"] is True


def test_gru_composite_fourier_loss_smoke(tmp_path: Path) -> None:
    data_root = create_synthetic_data_root(tmp_path)
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "smoke_gru_facl.yaml"
    config_path.write_text(
        f"""
experiment:
  name: smoke_gru_facl
  seed: 12
runtime:
  output_root: {output_root}
data:
  root: {data_root}
  source: series
  airports: [ZGSZ, ZGGG, VHHH, VMMC]
  target_airports: [ZGSZ]
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
loss:
  name: composite
  terms:
    - name: mse
      weight: 1.0
    - name: fourier_amplitude_correlation
      weight: 0.2
      params:
        mode: paper_random
        alpha: 0.1
model:
  name: gru
  hidden_size: 8
  num_layers: 1
  dropout: 0.0
trainer:
  device: cpu
  batch_size: 4
  epochs: 2
  patience: 2
  learning_rate: 0.001
  weight_decay: 0.0
  min_delta: 0.0
evaluation:
  metrics: [mae, rmse, bias]
  real_observation_only: true
""",
        encoding="utf-8",
    )

    train_result = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = Path(train_result.stdout.strip())
    assert (run_dir / "best_checkpoint.pt").is_file()
    assert (run_dir / "resolved_config.yaml").is_file()

    saved_config = (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    assert "fourier_amplitude_correlation" in saved_config


def test_gru_composite_patch_wise_structural_loss_smoke(tmp_path: Path) -> None:
    data_root = create_synthetic_data_root(tmp_path)
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "smoke_gru_psloss.yaml"
    config_path.write_text(
        f"""
experiment:
  name: smoke_gru_psloss
  seed: 14
runtime:
  output_root: {output_root}
data:
  root: {data_root}
  source: series
  airports: [ZGSZ, ZGGG, VHHH, VMMC]
  target_airports: [ZGSZ]
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
loss:
  name: composite
  terms:
    - name: mse
      weight: 1.0
    - name: patch_wise_structural
      weight: 3.0
      params:
        patch_len_threshold: 24
        mask_mode: strict_real_only
model:
  name: gru
  hidden_size: 8
  num_layers: 1
  dropout: 0.0
trainer:
  device: cpu
  batch_size: 4
  epochs: 2
  patience: 2
  learning_rate: 0.001
  weight_decay: 0.0
  min_delta: 0.0
evaluation:
  metrics: [mae, rmse, bias]
  real_observation_only: true
""",
        encoding="utf-8",
    )

    train_result = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = Path(train_result.stdout.strip())
    assert (run_dir / "best_checkpoint.pt").is_file()
    assert (run_dir / "resolved_config.yaml").is_file()

    saved_config = (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    assert "patch_wise_structural" in saved_config


def test_tfps_training_and_evaluation_smoke(tmp_path: Path) -> None:
    data_root = create_synthetic_data_root(tmp_path)
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "smoke_tfps.yaml"
    config_path.write_text(
        f"""
experiment:
  name: smoke_tfps
  seed: 13
runtime:
  output_root: {output_root}
data:
  root: {data_root}
  source: series
  airports: [ZGSZ, ZGGG, VHHH, VMMC]
  target_airports: [ZGSZ]
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
  name: tfps
  d_model: 8
  num_layers: 1
  n_heads: 2
  ff_dim: 16
  dropout: 0.0
  patch_len: 6
  stride: 6
  time_num_experts: 4
  time_top_k: 2
  frequency_num_experts: 4
  frequency_top_k: 2
  expert_hidden_size: 16
  subspace_eta: 5.0
  use_time_domain: true
  use_frequency_domain: true
  use_pattern_identifier: true
  use_pattern_experts: true
  noisy_gating: false
trainer:
  device: cpu
  batch_size: 4
  epochs: 1
  patience: 1
  learning_rate: 0.001
  weight_decay: 0.0
  min_delta: 0.0
evaluation:
  metrics: [mae, rmse, bias]
  real_observation_only: true
""",
        encoding="utf-8",
    )
    train_result = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = Path(train_result.stdout.strip())
    assert (run_dir / "best_checkpoint.pt").is_file()
    assert (run_dir / "resolved_config.yaml").is_file()

    eval_result = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--run-dir", str(run_dir)],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    metrics = json.loads(eval_result.stdout)
    assert set(metrics["test"]) == {"mae", "rmse", "bias"}


def test_timebridge_training_and_evaluation_smoke(tmp_path: Path) -> None:
    data_root = create_synthetic_data_root(tmp_path)
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "smoke_timebridge.yaml"
    config_path.write_text(
        f"""
experiment:
  name: smoke_timebridge
  seed: 17
runtime:
  output_root: {output_root}
data:
  root: {data_root}
  source: series
  airports: [ZGSZ, ZGGG, VHHH, VMMC]
  target_airports: [ZGSZ]
  input_variables:
    [
      sknt,
      wind_dir_sin,
      wind_dir_cos,
      tmpf,
      alti,
      dwpf,
      relh,
      gust,
      hour_sin,
      hour_cos,
      month_sin,
      month_cos,
      day_of_year_norm,
    ]
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
  name: timebridge
  period: 6
  num_p: 2
  ia_layers: 1
  pd_layers: 1
  ca_layers: 1
  stable_len: 6
  input_feature_count: 13
  shared_time_feature_count: 5
  d_model: 16
  n_heads: 4
  d_ff: 32
  dropout: 0.0
  attn_dropout: 0.1
  activation: gelu
trainer:
  device: cpu
  batch_size: 4
  epochs: 1
  patience: 1
  learning_rate: 0.001
  weight_decay: 0.0
  min_delta: 0.0
evaluation:
  metrics: [mae, rmse, bias]
  real_observation_only: true
""",
        encoding="utf-8",
    )
    train_result = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = Path(train_result.stdout.strip())
    assert (run_dir / "best_checkpoint.pt").is_file()
    assert (run_dir / "resolved_config.yaml").is_file()

    eval_result = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--run-dir", str(run_dir)],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    metrics = json.loads(eval_result.stdout)
    assert set(metrics["test"]) == {"mae", "rmse", "bias"}


def test_hcan_training_and_evaluation_smoke(tmp_path: Path) -> None:
    data_root = create_synthetic_data_root(tmp_path)
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "smoke_hcan.yaml"
    config_path.write_text(
        f"""
experiment:
  name: smoke_hcan
  seed: 23
runtime:
  output_root: {output_root}
data:
  root: {data_root}
  source: series
  airports: [ZGSZ, ZGGG, VHHH, VMMC]
  target_airports: [ZGSZ]
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
loss:
  name: composite
  terms:
    - name: hcan_auxiliary
      weight: 1.0
model:
  name: hcan
  backbone_hidden_size: 16
  backbone_num_layers: 1
  backbone_dropout: 0.0
  hidden_dim: 8
  num_coarse: 4
  num_fine: 8
  lambda_cls: 1.0
  lambda_reg: 1.0
  lambda_acl: 1.0
  lambda_direct: 1.0
trainer:
  device: cpu
  batch_size: 4
  epochs: 1
  patience: 1
  learning_rate: 0.001
  weight_decay: 0.0
  min_delta: 0.0
evaluation:
  metrics: [mae, rmse, bias]
  real_observation_only: true
""",
        encoding="utf-8",
    )

    train_result = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = Path(train_result.stdout.strip())
    assert (run_dir / "best_checkpoint.pt").is_file()
    assert (run_dir / "resolved_config.yaml").is_file()

    eval_result = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--run-dir", str(run_dir)],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    metrics = json.loads(eval_result.stdout)
    assert set(metrics["test"]) == {"mae", "rmse", "bias"}


def test_patchtst_series_15min_training_and_evaluation_smoke(tmp_path: Path) -> None:
    data_root = tmp_path / "synthetic_data"
    create_series_fixture(
        data_root / "series_15min",
        source="series_15min",
        shapes={"train": 224, "val": 208, "test": 208},
    )
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "smoke_patchtst_15min.yaml"
    config_path.write_text(
        f"""
experiment:
  name: smoke_patchtst_15min
  seed: 19
runtime:
  output_root: {output_root}
data:
  root: {data_root}
  source: series_15min
  airports: [ZGSZ, ZGGG, VHHH, VMMC]
  target_airports: [ZGSZ]
  input_variables:
    [
      sknt,
      wind_dir_sin,
      wind_dir_cos,
      tmpf,
      alti,
      dwpf,
      relh,
      gust,
      hour_sin,
      hour_cos,
      month_sin,
      month_cos,
      day_of_year_norm,
    ]
  target_variables: [sknt]
  time_resolution: 15min
  input_steps: 96
  forecast_steps: 96
normalization:
  enabled: true
  method: zscore
  fit_split: train
  apply_to_inputs: true
model:
  name: patchtst
  d_model: 16
  num_layers: 1
  n_heads: 4
  ff_dim: 32
  dropout: 0.0
  patch_len: 24
  stride: 12
trainer:
  device: cpu
  batch_size: 4
  epochs: 1
  patience: 1
  learning_rate: 0.001
  weight_decay: 0.0
  min_delta: 0.0
evaluation:
  metrics: [mae, rmse, bias]
  real_observation_only: true
""",
        encoding="utf-8",
    )

    train_result = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = Path(train_result.stdout.strip())
    assert (run_dir / "best_checkpoint.pt").is_file()
    assert (run_dir / "metrics.json").is_file()

    eval_result = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--run-dir", str(run_dir)],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    metrics = json.loads(eval_result.stdout)
    assert set(metrics["test"]) == {"mae", "rmse", "bias"}
    assert metrics["real_observation_only"] is True
