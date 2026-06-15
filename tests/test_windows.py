from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from windlab.config import load_config
from windlab.data.series import build_series_data
from windlab.data.windows import build_windowed_data

from .helpers import create_synthetic_data_root


def test_build_windows_keeps_splits_separate(tmp_path: Path) -> None:
    data_root = create_synthetic_data_root(tmp_path)
    os.environ["AIRPORT_WIND_DATA_ROOT"] = str(data_root)
    config = load_config("config/gru/baseline_hourly.yaml")

    prepared = build_series_data(config)
    assert prepared.train.values.shape == (72, 4, 13)
    assert prepared.train.targets.shape == (72, 1, 1)
    assert prepared.train.target_airport_ids == ["ZGSZ"]
    windowed = build_windowed_data(prepared, config)

    expected_train_samples = 72 - 24 - 24 + 1
    assert windowed.train.inputs.shape == (expected_train_samples, 24, 4, 13)
    assert windowed.train.targets.shape == (expected_train_samples, 24, 1, 1)
    assert windowed.val.sample_ids[0] == "val-00000"
    assert windowed.train.target_time_index[0, 0] == 24
    assert windowed.train.airport_ids == ["ZGSZ", "ZGGG", "VHHH", "VMMC"]
    assert windowed.train.target_airport_ids == ["ZGSZ"]
    assert np.all(windowed.train.observed_target_mask)


def test_build_windows_aligns_original_mask_for_series_15min(tmp_path: Path) -> None:
    data_root = create_synthetic_data_root(tmp_path)
    config_path = tmp_path / "series_15min.yaml"
    config_path.write_text(
        f"""
experiment:
  name: gru_series_15min
  seed: 7
runtime:
  output_root: outputs
data:
  root: {data_root}
  source: series_15min
  airports: [ZGSZ, ZGGG, VHHH, VMMC]
  target_airports: [ZGSZ]
  input_variables: [sknt]
  target_variables: [sknt]
  time_resolution: 15min
  input_steps: 4
  forecast_steps: 2
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
    config = load_config(config_path)
    prepared = build_series_data(config)
    windowed = build_windowed_data(prepared, config)
    expected_mask = np.array([True, False])
    assert np.array_equal(
        windowed.train.observed_target_mask[0, :, 0, 0],
        expected_mask,
    )
