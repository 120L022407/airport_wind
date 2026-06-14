from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from .helpers import create_synthetic_data_root


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
  input_variables: [sknt, wind_dir_sin, wind_dir_cos, tmpf, alti, dwpf, relh, gust, hour_sin, hour_cos, month_sin, month_cos, day_of_year_norm]
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
    metrics = json.loads(eval_result.stdout)
    assert set(metrics["validation"]) == {"mae", "rmse", "bias"}

    saved_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics["real_observation_only"] is True
