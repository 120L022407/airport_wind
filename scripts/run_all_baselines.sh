#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"

BASELINE_CONFIGS=(
  "config/gru/baseline_hourly.yaml"
  "config/patchtst/baseline_hourly.yaml"
  "config/itransformer/baseline_hourly.yaml"
  "config/dlinear/baseline_hourly.yaml"
  "config/tfps/baseline_hourly.yaml"
  "config/timebridge/baseline_hourly.yaml"
)

cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run evaluation: ${RUN_EVALUATION}"
if [[ -n "${AIRPORT_WIND_DATA_ROOT:-}" ]]; then
  echo "AIRPORT_WIND_DATA_ROOT: ${AIRPORT_WIND_DATA_ROOT}"
fi

for config_path in "${BASELINE_CONFIGS[@]}"; do
  echo
  echo "==> Training ${config_path}"
  run_dir="$("${PYTHON_BIN}" scripts/train.py --config "${config_path}" --output-root "${OUTPUT_ROOT}")"
  echo "Run directory: ${run_dir}"

  if [[ "${RUN_EVALUATION}" == "1" ]]; then
    echo "==> Evaluating ${run_dir}"
    "${PYTHON_BIN}" scripts/evaluate.py --run-dir "${run_dir}"
  fi
done
