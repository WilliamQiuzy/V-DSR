#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

process_num=""
for arg in "$@"; do
  case "$arg" in
    --process_num=*)
      process_num="${arg#*=}"
      ;;
  esac
done
if [[ -z "${process_num}" ]]; then
  for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--process_num" ]]; then
      j=$((i+1))
      process_num="${!j:-}"
      break
    fi
  done
fi
if [[ -z "${process_num}" ]]; then
  process_num="${PROCESS_NUM:-1}"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[GPU] nvidia-smi not found, run without CUDA_VISIBLE_DEVICES" >&2
  exec "${PYTHON_BIN}" "${ROOT_DIR}/src/data/qa_temp.py" "$@"
fi

mapfile -t gpu_lines < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
if [[ "${#gpu_lines[@]}" -eq 0 ]]; then
  echo "[GPU] no GPUs detected, run without CUDA_VISIBLE_DEVICES" >&2
  exec "${PYTHON_BIN}" "${ROOT_DIR}/src/data/qa_temp.py" "$@"
fi

gpu_ids="$(
  printf '%s\n' "${gpu_lines[@]}" \
    | sort -t, -k2 -nr \
    | head -n "${process_num}" \
    | awk -F',' '{gsub(/ /,"",$1); print $1}' \
    | paste -sd, -
)"

if [[ -z "${gpu_ids}" ]]; then
  echo "[GPU] no GPU selected, run without CUDA_VISIBLE_DEVICES" >&2
  exec "${PYTHON_BIN}" "${ROOT_DIR}/src/data/qa_temp.py" "$@"
fi

export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export PI3X_AUTO_GPU=0
export VDA_AUTO_GPU=0

echo "[GPU] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
exec "${PYTHON_BIN}" "${ROOT_DIR}/src/data/qa_temp.py" "$@"
