#!/usr/bin/env bash
set -euo pipefail

# MISATO inference: i10/i1 models, four recycling passes, and a 5-frame stitching prefix.
# Requires the inference dependencies listed in requirements.txt.
DRY_RUN=false
USE_MSA=true
while (( $# > 0 )); do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    --no-msa) USE_MSA=false ;;
    --msa) USE_MSA=true ;;
    -h|--help)
      echo "Usage: $0 [--msa|--no-msa] [--dry-run]"
      echo "Runs the MISATO i10/i1 models with MSA, N_cycle=4, and stitching prefix=5."
      echo "Use --no-msa for query-only sequence features. Model MSA sampling is capped at 16384 rows."
      echo "History lengths: 10 (coarse), 20 (fine); spatial interface crop: 256 tokens."
      echo "Sampling: 50 diffusion steps, 1 sample, noise scale=1.5, step scale=1.5; schedule=10:10;10:1."
      echo "ATMOS_ROOT_DIR defaults to atmos_oss/ in the repository; PROTENIX_ROOT_DIR defaults to repository af3-dev/."
      echo "Other overrides: PYTHON_BIN, CHECKPOINT_DIR, CONFIG_DIR, OUTPUT_ROOT_DIR, RUN_ID_PREFIX, SOURCE_INDICES_CSV,"
      echo "                 GPUS_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, DATA_NUM_DL_WORKERS,"
      echo "                 BIOASSEMBLY_DIR, MSA_DIR, MSA_INDEX, TRIANGLE_ATTENTION, TRIANGLE_MULTIPLICATIVE, LOAD_STRICT"
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
CONFIG_DIR="${CONFIG_DIR:-${CHECKPOINT_DIR}}"
OUT_DIR="${OUTPUT_ROOT_DIR:-${REPO_DIR}/output/misato_inference}"
MSA_TAG=msa
if ! "${USE_MSA}"; then MSA_TAG=no_msa; fi
RUN_ID_PREFIX="${RUN_ID_PREFIX:-misato_ncycle4_${MSA_TAG}}"
SOURCE_INDICES_CSV="${SOURCE_INDICES_CSV:-${ATMOS_ROOT_DIR}/misato/test_MD.csv}"
BIOASSEMBLY_DIR="${BIOASSEMBLY_DIR:-${ATMOS_ROOT_DIR}/misato/test_MD}"
MSA_DIR="${MSA_DIR:-${ATMOS_ROOT_DIR}/misato/msa_template}"
MSA_INDEX="${MSA_INDEX:-${ATMOS_ROOT_DIR}/misato/seq_to_pdb_index.json}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${MLP_WORKER_GPU:-1}}"
NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-1234}}"
DATA_NUM_DL_WORKERS="${DATA_NUM_DL_WORKERS:-2}"
LOAD_STRICT="${LOAD_STRICT:-false}"

for file in \
  "${CHECKPOINT_DIR}/misato_i10.pt" \
  "${CHECKPOINT_DIR}/misato_i1.pt" \
  "${CONFIG_DIR}/misato_i10_config.yaml" \
  "${CONFIG_DIR}/misato_i1_config.yaml" \
  "${SOURCE_INDICES_CSV}"; do
  if [[ ! -s "${file}" ]]; then
    echo "Missing input: ${file}" >&2
    exit 1
  fi
done
if [[ ! -d "${BIOASSEMBLY_DIR}" ]]; then
  echo "Missing trajectory directory: ${BIOASSEMBLY_DIR}" >&2
  exit 1
fi
if "${USE_MSA}" && { [[ ! -d "${MSA_DIR}" ]] || [[ ! -s "${MSA_INDEX}" ]]; }; then
  echo "Missing MSA directory or index: ${MSA_DIR}; ${MSA_INDEX}" >&2
  exit 1
fi

checkpoint_path="${CHECKPOINT_DIR}/misato_i10.pt;${CHECKPOINT_DIR}/misato_i1.pt"
load_config_path="${CONFIG_DIR}/misato_i10_config.yaml;${CONFIG_DIR}/misato_i1_config.yaml"
stage_mod="model.N_cycle,4,model.msa_module.msa_max_size,16384,triangle_attention,${TRIANGLE_ATTENTION},triangle_multiplicative,${TRIANGLE_MULTIPLICATIVE},sample_diffusion.N_sample,1,sample_diffusion.N_step,50,sample_diffusion.noise_scale_lambda,1.5,sample_diffusion.step_scale_eta,1.5,data.msa.enable_prot_msa,${USE_MSA},data.msa.sample_cutoff.train,16384,data.msa.sample_cutoff.test,16384"
load_config_mod="${stage_mod},model_max_track_length,10,destination_enable,false;${stage_mod},model_max_track_length,20,destination_enable,true"

command=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nproc_per_node "${GPUS_PER_NODE}" --nnodes "${NNODES}" --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}"
  "${REPO_DIR}/atmos/runner/md_inference.py"
  --model_name protenix_base_default_v0.5.0
  --run_name "${RUN_ID_PREFIX}" --seed 42 --base_dir "${OUT_DIR}"
  --dtype bf16 --project atmos_inference --use_wandb false
  --eval_interval 500 --log_interval 10 --max_steps 10000 --train_crop_size 256
  --model.N_cycle 4 --model.msa_module.msa_max_size 16384
  --sample_diffusion.N_sample 1 --sample_diffusion.N_step 50
  --sample_diffusion.noise_scale_lambda 1.5 --sample_diffusion.step_scale_eta 1.5
  --load_checkpoint_path "${checkpoint_path}" --load_config_path "${load_config_path}"
  --load_config_mod "${load_config_mod}" --load_strict "${LOAD_STRICT}"
  --data.train_sets misato_test --data.test_sets misato_test
  --data.msa.enable_prot_msa "${USE_MSA}"
  --data.msa.sample_cutoff.train 16384 --data.msa.sample_cutoff.test 16384
  --data.misato_test.msa.enable_prot_msa "${USE_MSA}"
  --data.misato_test.msa.prot_msadir_raw_paths "${MSA_DIR}"
  --data.misato_test.msa.prot_seq_or_filename_to_msadir_jsons "${MSA_INDEX}"
  --data.misato_test.base_info.indices_fpath "${SOURCE_INDICES_CSV}"
  --data.misato_test.base_info.bioassembly_dict_dir "${BIOASSEMBLY_DIR}"
  --infer_trajectory_schedule "10:10;10:1"
  --infer_trajectory_mode long_traj:multiple_model_multiple_interval_w_stitching
  --largest_batch_latent_num 5 --data.num_dl_workers "${DATA_NUM_DL_WORKERS}"
  --infer_trajectory_stitching_prefix_num 5
)

echo "Inference: N_cycle=4, MSA=${USE_MSA}, MSA cutoff=16384, stitching prefix=5, history=10/20, models=misato_i10/misato_i1"
echo "Sampling: N_step=50, N_sample=1, noise_scale_lambda=1.5, step_scale_eta=1.5, schedule=10:10;10:1, seed=42, dtype=bf16"
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
