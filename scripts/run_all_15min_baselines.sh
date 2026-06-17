#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

BASELINE_CONFIGS=(
  "config/gru/baseline_15min.yaml"
  "config/patchtst/baseline_15min.yaml"
  "config/itransformer/baseline_15min.yaml"
  "config/dlinear/baseline_15min.yaml"
  "config/tfps/baseline_15min.yaml"
  "config/timebridge/baseline_15min.yaml"
)

cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Output root: ${OUTPUT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
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
