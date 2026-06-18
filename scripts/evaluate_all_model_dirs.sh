#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/outputs/launcher_logs}"
LAUNCH_TAG="${LAUNCH_TAG:-evaluate_all}"
TARGET_DIR="${1:-}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/evaluate_all_model_dirs.sh <target_dir>

Description:
  Evaluate all saved run directories under <target_dir> using the unified
  scripts/evaluate.py entry point.

Supported layouts:
  1. <target_dir>/<run_dir>
  2. <target_dir>/<model_dir>/<run_dir>

A run directory is detected by:
  - resolved_config.yaml
  - and one of: best_checkpoint.pt / last_checkpoint.pt / checkpoint.pt
EOF
}

is_run_dir() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 1
  [[ -f "${dir}/resolved_config.yaml" ]] || return 1
  [[ -f "${dir}/best_checkpoint.pt" ]] \
    || [[ -f "${dir}/last_checkpoint.pt" ]] \
    || [[ -f "${dir}/checkpoint.pt" ]]
}

collect_run_dirs() {
  local root_dir="$1"
  local child_dir
  local grandchild_dir

  if is_run_dir "${root_dir}"; then
    RUN_DIRS+=("${root_dir}")
    return 0
  fi

  shopt -s nullglob
  for child_dir in "${root_dir}"/*; do
    [[ -d "${child_dir}" ]] || continue
    if is_run_dir "${child_dir}"; then
      RUN_DIRS+=("${child_dir}")
      continue
    fi
    for grandchild_dir in "${child_dir}"/*; do
      [[ -d "${grandchild_dir}" ]] || continue
      if is_run_dir "${grandchild_dir}"; then
        RUN_DIRS+=("${grandchild_dir}")
      fi
    done
  done
  shopt -u nullglob
}

if [[ -z "${TARGET_DIR}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "Target directory does not exist: ${TARGET_DIR}" >&2
  exit 1
fi

TARGET_DIR="$(cd "${TARGET_DIR}" && pwd)"
TARGET_NAME="$(basename "${TARGET_DIR}")"
LAUNCH_DIR="${LOG_ROOT%/}/${LAUNCH_TAG}_${TARGET_NAME}_${TIMESTAMP}"

declare -a RUN_DIRS=()
declare -a FAILED_RUNS=()
declare -i SUCCESS_COUNT=0

collect_run_dirs "${TARGET_DIR}"

if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
  echo "No saved run directories found under: ${TARGET_DIR}" >&2
  exit 1
fi

mkdir -p "${LAUNCH_DIR}"
cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Target dir: ${TARGET_DIR}"
echo "Found run dirs: ${#RUN_DIRS[@]}"
echo "Log dir: ${LAUNCH_DIR}"

for run_dir in "${RUN_DIRS[@]}"; do
  run_name="$(basename "${run_dir}")"
  parent_name="$(basename "$(dirname "${run_dir}")")"
  log_stem="${run_name}"
  if [[ "$(dirname "${run_dir}")" != "${TARGET_DIR}" ]]; then
    log_stem="${parent_name}_${run_name}"
  fi
  log_path="${LAUNCH_DIR}/${log_stem}.log"

  echo
  echo "==> Evaluating ${run_dir}"
  if "${PYTHON_BIN}" scripts/evaluate.py --run-dir "${run_dir}" \
    > "${log_path}" 2>&1; then
    SUCCESS_COUNT+=1
    echo "[ok] ${run_dir}"
    echo "     log: ${log_path}"
  else
    FAILED_RUNS+=("${run_dir}")
    echo "[failed] ${run_dir}" >&2
    echo "         log: ${log_path}" >&2
  fi
done

echo
echo "Evaluation finished."
echo "Succeeded: ${SUCCESS_COUNT}"
echo "Failed: ${#FAILED_RUNS[@]}"
echo "Logs: ${LAUNCH_DIR}"

if [[ "${#FAILED_RUNS[@]}" -gt 0 ]]; then
  echo "Failed run directories:" >&2
  for run_dir in "${FAILED_RUNS[@]}"; do
    echo "  - ${run_dir}" >&2
  done
  exit 1
fi
