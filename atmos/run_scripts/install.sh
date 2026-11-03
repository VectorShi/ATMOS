#!/usr/bin/env bash
set -euo pipefail

if (( $# > 0 )); then
  if [[ $# == 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    echo "Usage: bash atmos/run_scripts/install.sh"
    echo "Installs ATMOS dependencies into the active Python 3.11 environment."
    echo "Requires Linux and a CUDA-capable NVIDIA GPU for inference."
    exit 0
  fi
  echo "Usage: bash atmos/run_scripts/install.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit("Use a Python 3.11 environment; see docs/installation.md.")
PY

# DeepSpeed's package setup requires PyTorch to be installed first.
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
DS_BUILD_OPS=0 python -m pip install -r requirements.txt 'modelscope-hub>=0.5.0'
python - <<'PY'
import torch
print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; GPU available: {torch.cuda.is_available()}")
PY
