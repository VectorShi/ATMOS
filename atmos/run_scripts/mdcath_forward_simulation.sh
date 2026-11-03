#!/usr/bin/env bash
set -euo pipefail

# mdCATH inference: MSA, 4 recycling passes, and a 5-frame stitching prefix.
# Requires the inference dependencies listed in requirements.txt.
DRY_RUN=false
while (( $# > 0 )); do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    -h|--help)
      echo "Usage: $0 [--dry-run]"
      echo "Runs the mdCATH i20/i1 models with MSA, N_cycle=4, and stitching prefix=5."
      echo "History lengths: 20 (coarse), 30 (fine)."
      echo "ATMOS_ROOT_DIR defaults to atmos_oss/ in the repository; PROTENIX_ROOT_DIR defaults to repository af3-dev/."
      echo "Other overrides: PYTHON_BIN, CHECKPOINT_DIR, OUTPUT_ROOT_DIR, RUN_ID_PREFIX, SOURCE_INDICES_CSV,"
      echo "                 GPUS_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, DATA_NUM_DL_WORKERS,"
      echo "                 MSA_DIR, MSA_INDEX, TRIANGLE_ATTENTION, TRIANGLE_MULTIPLICATIVE, LOAD_STRICT"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LAYERNORM_TYPE=torch
export ATMOS_ROOT_DIR="${ATMOS_ROOT_DIR:-${REPO_DIR}/atmos_oss}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-${REPO_DIR}/af3-dev}"
export NCCL_TIMEOUT_SECOND="${NCCL_TIMEOUT_SECOND:-1800}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TRIANGLE_ATTENTION="${TRIANGLE_ATTENTION:-triattention}"
export TRIANGLE_MULTIPLICATIVE="${TRIANGLE_MULTIPLICATIVE:-cuequivariance}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ATMOS_ROOT_DIR}/atmos_checkpoints}"
OUT_DIR="${OUTPUT_ROOT_DIR:-${REPO_DIR}/output/mdcath_inference}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-mdcath_ncycle4_msa}"
SOURCE_INDICES_CSV="${SOURCE_INDICES_CSV:-${ATMOS_ROOT_DIR}/mdCATH_tempo/mdCATH_tempo_test.csv}"
MSA_DIR="${MSA_DIR:-${ATMOS_ROOT_DIR}/mdCATH_tempo/mdcath_msa/msa_template}"
MSA_INDEX="${MSA_INDEX:-${ATMOS_ROOT_DIR}/mdCATH_tempo/mdcath_msa/seq_to_pdb_index.json}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${MLP_WORKER_GPU:-1}}"
NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-1234}}"
DATA_NUM_DL_WORKERS="${DATA_NUM_DL_WORKERS:-2}"
LOAD_STRICT="${LOAD_STRICT:-false}"

for file in \
  "${CHECKPOINT_DIR}/mdCATH_i20.pt" \
  "${CHECKPOINT_DIR}/mdCATH_i1.pt" \
  "${CHECKPOINT_DIR}/mdCATH_i20_config.yaml" \
  "${CHECKPOINT_DIR}/mdCATH_i1_config.yaml" \
  "${SOURCE_INDICES_CSV}" "${MSA_INDEX}"; do
  if [[ ! -s "${file}" ]]; then
    echo "Missing input: ${file}" >&2
    exit 1
  fi
done
if [[ ! -d "${MSA_DIR}" ]]; then
  echo "Missing MSA directory: ${MSA_DIR}" >&2
  exit 1
fi

checkpoint_path="${CHECKPOINT_DIR}/mdCATH_i20.pt;${CHECKPOINT_DIR}/mdCATH_i1.pt"
load_config_path="${CHECKPOINT_DIR}/mdCATH_i20_config.yaml;${CHECKPOINT_DIR}/mdCATH_i1_config.yaml"
stage_mod="model.N_cycle,4,triangle_attention,${TRIANGLE_ATTENTION},triangle_multiplicative,${TRIANGLE_MULTIPLICATIVE},sample_diffusion.N_sample,1,sample_diffusion.N_step,50,sample_diffusion.noise_scale_lambda,1.5,sample_diffusion.step_scale_eta,1.5"
load_config_mod="${stage_mod},model_max_track_length,20;${stage_mod},model_max_track_length,30"

command=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nproc_per_node "${GPUS_PER_NODE}" --nnodes "${NNODES}" --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}"
  "${REPO_DIR}/atmos/runner/md_inference.py"
  --run_name "${RUN_ID_PREFIX}" --seed 42 --base_dir "${OUT_DIR}"
  --dtype bf16 --project atmos_inference --use_wandb false
  --eval_interval 500 --log_interval 10 --max_steps 10000 --train_crop_size 256
  --model.N_cycle 4
  --load_checkpoint_path "${checkpoint_path}" --load_config_path "${load_config_path}"
  --load_config_mod "${load_config_mod}" --load_strict "${LOAD_STRICT}"
  --data.train_sets mdCATH_tempo_test --data.test_sets mdCATH_tempo_test
  --data.msa.enable_prot_msa true
  --data.mdCATH_tempo_test.msa.enable_prot_msa true
  --data.mdCATH_tempo_test.msa.prot_msadir_raw_paths "${MSA_DIR}"
  --data.mdCATH_tempo_test.msa.prot_seq_or_filename_to_msadir_jsons "${MSA_INDEX}"
  --data.mdCATH_tempo_test.base_info.indices_fpath "${SOURCE_INDICES_CSV}"
  --infer_trajectory_schedule "20:20;20:1"
  --infer_trajectory_mode long_traj:multiple_model_multiple_interval_w_stitching
  --largest_batch_latent_num 5 --data.num_dl_workers "${DATA_NUM_DL_WORKERS}"
  --infer_trajectory_stitching_prefix_num 5
)

echo "Inference: N_cycle=4, MSA=true, stitching prefix=5, history=20/30, models=mdCATH_i20/mdCATH_i1"
echo "Output root: ${OUT_DIR}; run prefix: ${RUN_ID_PREFIX}"
if "${DRY_RUN}"; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUT_DIR}/atmos_log"
OUTPUT_LOG_FILE="${OUT_DIR}/atmos_log/${RUN_ID_PREFIX}_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "Log: ${OUTPUT_LOG_FILE}"
ulimit -n 4096
exec "${command[@]}" > "${OUTPUT_LOG_FILE}" 2>&1
