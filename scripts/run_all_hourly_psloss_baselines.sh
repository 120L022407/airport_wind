#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
GPU_IDS_RAW="${GPU_IDS:-0}"
LAUNCH_TAG="${LAUNCH_TAG:-hourly_psloss}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LAUNCH_DIR="${PROJECT_ROOT}/${OUTPUT_ROOT}/launcher_logs/${LAUNCH_TAG}_${TIMESTAMP}"

IFS=', ' read -r -a GPU_IDS <<< "${GPU_IDS_RAW}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "GPU_IDS must not be empty." >&2
  exit 1
fi

BASELINE_CONFIGS=(
  "config/gru/baseline_hourly_psloss.yaml"
  "config/patchtst/baseline_hourly_psloss.yaml"
  "config/itransformer/baseline_hourly_psloss.yaml"
  "config/dlinear/baseline_hourly_psloss.yaml"
  "config/tfps/baseline_hourly_psloss.yaml"
  "config/timebridge/baseline_hourly_psloss.yaml"
  "config/hcan/baseline_hourly_psloss.yaml"
)

mkdir -p "${LAUNCH_DIR}"
cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run evaluation: ${RUN_EVALUATION}"
echo "GPU_IDS: ${GPU_IDS[*]}"
echo "Launch dir: ${LAUNCH_DIR}"
if [[ -n "${AIRPORT_WIND_DATA_ROOT:-}" ]]; then
  echo "AIRPORT_WIND_DATA_ROOT: ${AIRPORT_WIND_DATA_ROOT}"
fi

for index in "${!BASELINE_CONFIGS[@]}"; do
  config_path="${BASELINE_CONFIGS[${index}]}"
  gpu_id="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
  model_name="$(basename "$(dirname "${config_path}")")"
  config_name="$(basename "${config_path}" .yaml)"
  run_name="${model_name}_${config_name}"
  log_path="${LAUNCH_DIR}/${run_name}.log"
  pid_path="${LAUNCH_DIR}/${run_name}.pid"

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    cd "${PROJECT_ROOT}"

    echo "[start] $(date '+%F %T') config=${config_path} gpu=${gpu_id}"
    run_dir="$("${PYTHON_BIN}" scripts/train.py --config "${config_path}" --output-root "${OUTPUT_ROOT}")"
    echo "[run_dir] ${run_dir}"

    if [[ "${RUN_EVALUATION}" == "1" ]]; then
      echo "[eval] $(date '+%F %T') run_dir=${run_dir}"
      "${PYTHON_BIN}" scripts/evaluate.py --run-dir "${run_dir}"
    fi

    echo "[done] $(date '+%F %T') config=${config_path}"
  ) > "${log_path}" 2>&1 &

  pid="$!"
  echo "${pid}" > "${pid_path}"
  echo "Launched ${config_path} on GPU ${gpu_id}: pid=${pid}, log=${log_path}"
done

echo
echo "All jobs launched in background."
echo "Logs and pid files: ${LAUNCH_DIR}"
echo "Example: tail -f ${LAUNCH_DIR}/gru_baseline_hourly_psloss.log"
