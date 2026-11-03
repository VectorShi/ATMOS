#!/usr/bin/env bash
set -euo pipefail

# Run two mdCATH index groups through the same inference entry point.
# Both groups use GPUs 0-7 by default, with two inference processes per GPU.
DRY_RUN=false
while (( $# > 0 )); do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    -h|--help)
      echo "Usage: $0 [--dry-run]"
      echo "Runs two groups with MSA, N_cycle=4, and stitching prefix=5."
      echo "Both groups use all visible GPUs (default: 0-7)."
      echo "Overrides: GPU_IDS_PART0, GPU_IDS_PART1, MASTER_PORT_PART0, MASTER_PORT_PART1,"
      echo "           and the environment variables supported by mdcath_forward_simulation.sh."
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export ATMOS_ROOT_DIR="${ATMOS_ROOT_DIR:-${REPO_DIR}/atmos_oss}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUTPUT_ROOT_DIR:-${REPO_DIR}/output/mdcath_inference}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-mdcath_ncycle4_msa}"
LAUNCH_DIR="${OUT_DIR}/${RUN_ID_PREFIX}_$(date +%Y%m%d_%H%M%S)_$$"
SOURCE_INDICES_CSV="${SOURCE_INDICES_CSV:-${ATMOS_ROOT_DIR}/mdCATH_tempo/mdCATH_tempo_test.csv}"
GPU_IDS_PART0="${GPU_IDS_PART0:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
GPU_IDS_PART1="${GPU_IDS_PART1:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
MASTER_PORT_PART0="${MASTER_PORT_PART0:-29502}"
MASTER_PORT_PART1="${MASTER_PORT_PART1:-29503}"
if [[ "${MASTER_PORT_PART0}" == "${MASTER_PORT_PART1}" ]]; then
  echo "The two torchrun jobs require different master ports." >&2
  exit 2
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/split_mdcath_tempo_indices.py" \
  --input-csv "${SOURCE_INDICES_CSV}" \
  --part0-csv "${LAUNCH_DIR}/indices/part0.csv" \
  --part1-csv "${LAUNCH_DIR}/indices/part1.csv" --pdb-id-suffix "_0"

run_part() {
  local part="$1" gpu_ids="$2" port="$3"
  local -a gpu_list extra_args=()
  IFS=',' read -r -a gpu_list <<< "${gpu_ids}"
  if "${DRY_RUN}"; then
    extra_args+=(--dry-run)
  fi
  echo "${part}: GPUs=${gpu_ids}, master_port=${port}"
  env CUDA_VISIBLE_DEVICES="${gpu_ids}" GPUS_PER_NODE="${#gpu_list[@]}" \
    NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT="${port}" \
    PYTHON_BIN="${PYTHON_BIN}" OUTPUT_ROOT_DIR="${LAUNCH_DIR}" \
    RUN_ID_PREFIX="${RUN_ID_PREFIX}_${part}" \
    SOURCE_INDICES_CSV="${LAUNCH_DIR}/indices/${part}.csv" \
    bash "${SCRIPT_DIR}/mdcath_forward_simulation.sh" "${extra_args[@]}"
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
trap 'kill -TERM "$pid0" "$pid1" 2>/dev/null || true' INT TERM
status=0
wait "${pid0}" || status=$?
wait "${pid1}" || { part1_status=$?; if (( status == 0 )); then status="${part1_status}"; fi; }
exit "${status}"
