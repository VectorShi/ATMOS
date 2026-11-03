#!/usr/bin/env bash
set -euo pipefail

# MISATO training with protein MSA and a frozen 48-layer Pairformer.
DRY_RUN=false
while (( $# > 0 )); do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    -h|--help)
      echo "Usage: $0 [--dry-run]"
      echo "Trains the 10-frame interval model: N_cycle=4, history=10, destination=false, MSA=true."
      echo "Uses misato_train for training and misato_valid for validation."
      echo "Initializes from Protenix v0.5.0 and freezes its embedding modules and Pairformer."
      echo "ATMOS_ROOT_DIR defaults to atmos_oss/ in the repository; PROTENIX_ROOT_DIR defaults to repository af3-dev/."
      echo "Other overrides: PYTHON_BIN, INIT_CHECKPOINT_PATH, OUTPUT_ROOT_DIR, GPUS_PER_NODE (default: 8),"
      echo "                 NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, TRIANGLE_ATTENTION, TRIANGLE_MULTIPLICATIVE"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=offline
export LAYERNORM_TYPE=torch
export ATMOS_ROOT_DIR="${ATMOS_ROOT_DIR:-${REPO_DIR}/atmos_oss}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-${REPO_DIR}/af3-dev}"
export NCCL_TIMEOUT_SECOND="${NCCL_TIMEOUT_SECOND:-1800}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TRIANGLE_ATTENTION="${TRIANGLE_ATTENTION:-triattention}"
export TRIANGLE_MULTIPLICATIVE="${TRIANGLE_MULTIPLICATIVE:-cuequivariance}"

PYTHON_BIN="${PYTHON_BIN:-python}"
checkpoint_path="${INIT_CHECKPOINT_PATH:-${PROTENIX_ROOT_DIR}/checkpoint/protenix_base_default_v0.5.0.pt}"
OUT_DIR="${OUTPUT_ROOT_DIR:-output}"
run_name="misato_i10"
GPUS_PER_NODE="${GPUS_PER_NODE:-${MLP_WORKER_GPU:-8}}"
NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-1234}}"

command=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nproc_per_node "${GPUS_PER_NODE}" --nnodes "${NNODES}" --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}"
  "${REPO_DIR}/atmos/runner/md_train.py"
  --model_name protenix_base_default_v0.5.0
  --run_name "${run_name}" --seed 42 --base_dir "${OUT_DIR}"
  --dtype bf16 --project atmos --use_wandb false
  --eval_interval 500 --log_interval 10 --checkpoint_interval 1000
  --ema_decay 0.999 --train_crop_size 256
  --max_steps 100000 --warmup_steps 2000 --lr 0.0005
  --model.N_cycle 4 --model.pairformer.n_blocks 48 --model.msa_module.msa_max_size 1024
  --model_max_track_length 10 --destination_enable false
  --freeze_pairformer true --prefix_batch_size 6
  --triangle_attention "${TRIANGLE_ATTENTION}" --triangle_multiplicative "${TRIANGLE_MULTIPLICATIVE}"
  --load_checkpoint_path "${checkpoint_path}" --load_ema_checkpoint_path "${checkpoint_path}"
  --load_strict false
  --loss.weight.alpha_confidence 0.0 --loss.weight.alpha_bond 0.0 --loss.weight.smooth_lddt 0.0
  --data.train_sets misato_train --data.test_sets misato_valid
  --data.msa.enable_prot_msa true
  --data.msa.sample_cutoff.train 1024 --data.msa.sample_cutoff.test 1024
  --data.misato_train.msa.enable_prot_msa true
  --data.misato_valid.msa.enable_prot_msa true
  --data.misato_train.sample_subsequence_configs.interval_min 10
  --data.misato_train.sample_subsequence_configs.interval_max 10
  --data.misato_train.sample_subsequence_configs.max_len 10
  --data.misato_valid.sample_subsequence_configs.interval_min 10
  --data.misato_valid.sample_subsequence_configs.interval_max 10
  --data.misato_valid.sample_subsequence_configs.max_len 10
  --diffusion_batch_size 8 --diffusion_chunk_size 1
  --infer_setting.sample_diffusion_chunk_size 5
  --sample_diffusion.N_sample 5 --sample_diffusion.N_step 50
  --sample_diffusion.noise_scale_lambda 1.003 --sample_diffusion.step_scale_eta 1.5
)

echo "Training: interval=10 frames, N_cycle=4, history=10, destination=false, MSA=true, frozen Pairformer"
echo "Datasets: train=misato_train; validation=misato_valid"
if "${DRY_RUN}"; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

for file in \
  "${checkpoint_path}" \
  "${ATMOS_ROOT_DIR}/misato/train_MD.csv" \
  "${ATMOS_ROOT_DIR}/misato/val_MD.csv" \
  "${ATMOS_ROOT_DIR}/misato/seq_to_pdb_index.json"; do
  if [[ ! -s "${file}" ]]; then
    echo "Missing input: ${file}" >&2
    exit 1
  fi
done

for directory in \
  "${ATMOS_ROOT_DIR}/misato/train_MD" \
  "${ATMOS_ROOT_DIR}/misato/val_MD" \
  "${ATMOS_ROOT_DIR}/misato/msa_template"; do
  if [[ ! -d "${directory}" ]]; then
    echo "Missing input directory: ${directory}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_DIR}/atmos_log"
OUTPUT_LOG_FILE="${OUT_DIR}/atmos_log/${run_name}_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "Log: ${OUTPUT_LOG_FILE}"
ulimit -n 4096
exec "${command[@]}" > "${OUTPUT_LOG_FILE}" 2>&1
