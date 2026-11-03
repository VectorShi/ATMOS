#!/usr/bin/env bash
set -euo pipefail

# Run two groups of MISATO complexes concurrently through the inference entry point.
# Both groups use GPUs 0-7 by default, with two inference processes per GPU.
DRY_RUN=false
MSA_FLAG=--msa
while (( $# > 0 )); do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    --msa|--no-msa) MSA_FLAG="$1" ;;
    -h|--help)
      echo "Usage: $0 [--msa|--no-msa] [--dry-run]"
      echo "Splits the MISATO test complexes into two groups, preserving all interface rows per complex."
      echo "Both groups use all visible GPUs (default: 0-7), with separate torchrun master ports."
      echo "Uses misato_forward_simulation.sh: MSA enabled, N_cycle=4, 50 diffusion steps, stitching prefix=5."
      echo "--dry-run writes the two index CSVs and prints commands without starting inference."
      echo "Overrides: GPU_IDS_PART0, GPU_IDS_PART1, MASTER_PORT_PART0, MASTER_PORT_PART1,"
      echo "           and the asset/output variables supported by misato_forward_simulation.sh."
      echo "Each group runs on one node; its GPU count is determined by its GPU ID list."
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export ATMOS_ROOT_DIR="${ATMOS_ROOT_DIR:-${REPO_DIR}/atmos_oss}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUTPUT_ROOT_DIR:-${REPO_DIR}/output/misato_inference}"
MSA_TAG=msa
if [[ "${MSA_FLAG}" == --no-msa ]]; then MSA_TAG=no_msa; fi
RUN_ID_PREFIX="${RUN_ID_PREFIX:-misato_ncycle4_${MSA_TAG}}"
LAUNCH_DIR="${OUT_DIR}/${RUN_ID_PREFIX}_$(date +%Y%m%d_%H%M%S)_$$"
SOURCE_INDICES_CSV="${SOURCE_INDICES_CSV:-${ATMOS_ROOT_DIR}/misato/test_MD.csv}"
GPU_IDS_PART0="${GPU_IDS_PART0:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
GPU_IDS_PART1="${GPU_IDS_PART1:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
MASTER_PORT_PART0="${MASTER_PORT_PART0:-29512}"
MASTER_PORT_PART1="${MASTER_PORT_PART1:-29513}"
if [[ "${MASTER_PORT_PART0}" == "${MASTER_PORT_PART1}" ]]; then
  echo "The two torchrun jobs require different master ports." >&2
  exit 2
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/split_misato_indices.py" \
  --input-csv "${SOURCE_INDICES_CSV}" \
  --part0-csv "${LAUNCH_DIR}/indices/part0.csv" \
  --part1-csv "${LAUNCH_DIR}/indices/part1.csv"

run_part() {
  local part="$1" gpu_ids="$2" port="$3"
  local -a gpu_list extra_args=("${MSA_FLAG}") command
  IFS=',' read -r -a gpu_list <<< "${gpu_ids}"
  if "${DRY_RUN}"; then extra_args+=(--dry-run); fi
  echo "${part}: GPUs=${gpu_ids}, master_port=${port}"
  command=(env CUDA_VISIBLE_DEVICES="${gpu_ids}" GPUS_PER_NODE="${#gpu_list[@]}"
    NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT="${port}"
    PYTHON_BIN="${PYTHON_BIN}" OUTPUT_ROOT_DIR="${LAUNCH_DIR}"
    RUN_ID_PREFIX="${RUN_ID_PREFIX}_${part}"
    SOURCE_INDICES_CSV="${LAUNCH_DIR}/indices/${part}.csv"
    bash "${SCRIPT_DIR}/misato_forward_simulation.sh" "${extra_args[@]}")
  if "${DRY_RUN}"; then
    "${command[@]}"
  else
    exec "${command[@]}"
  fi
}

echo "Outputs and logs: ${LAUNCH_DIR}"
if "${DRY_RUN}"; then
  run_part part0 "${GPU_IDS_PART0}" "${MASTER_PORT_PART0}"
  run_part part1 "${GPU_IDS_PART1}" "${MASTER_PORT_PART1}"
  exit 0
fi

run_part part0 "${GPU_IDS_PART0}" "${MASTER_PORT_PART0}" > "${LAUNCH_DIR}/part0.log" 2>&1 &
pid0=$!
run_part part1 "${GPU_IDS_PART1}" "${MASTER_PORT_PART1}" > "${LAUNCH_DIR}/part1.log" 2>&1 &
pid1=$!
stop_parts() {
  trap '' INT TERM
  kill -TERM "${pid0}" "${pid1}" 2>/dev/null || true
  wait "${pid0}" 2>/dev/null || true
  wait "${pid1}" 2>/dev/null || true
  exit "$1"
}
trap 'stop_parts 130' INT
trap 'stop_parts 143' TERM
status=0
wait "${pid0}" || status=$?
wait "${pid1}" || { part1_status=$?; if (( status == 0 )); then status="${part1_status}"; fi; }
exit "${status}"
