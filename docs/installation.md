# Installation

## Environment setup

Use Linux, Python 3.11, and an NVIDIA GPU with a driver compatible with the
CUDA-enabled PyTorch 2.7.1 packages in [requirements.txt](../requirements.txt).
The mdCATH quick-start example has been run on a 24 GB RTX 4090; memory needs
increase with system size. Install Git, wget, GNU tar, a C++ build toolchain,
and Conda before running the commands below.

```bash
conda create -n atmos python=3.11 -y
conda activate atmos
bash atmos/run_scripts/install.sh
```

The installer uses the active Python environment. It installs PyTorch before
DeepSpeed and installs the ModelScope Hub client for asset downloads. Run
commands from the repository root; the launchers make the local code available
on `PYTHONPATH`.

Check that the environment can access the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip check
```

The inference scripts use `triattention` and `cuequivariance` by default and
set `LAYERNORM_TYPE=torch`. To use PyTorch attention kernels, set
`TRIANGLE_ATTENTION=torch TRIANGLE_MULTIPLICATIVE=torch` when launching.

### Optional CUTLASS kernels

The default ATMOS launchers do not require CUTLASS. For DeepSpeed/ds4sci
attention kernels, install CUTLASS v3.5.1 and export its path in the launch shell:

```bash
git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git cutlass
export CUTLASS_PATH="$PWD/cutlass"
```

## Quick-start assets

```bash
bash atmos/run_scripts/download_example.sh
```

This downloads the mdCATH i20/i1 weights and YAMLs, the mdCATH test trajectory
archive, its MSA archive and sequence index, and Protenix common files. It also
writes `atmos_oss/examples/mdcath.csv`, containing the 53-token test protein
`2hsna01`. The input includes its starting structure and precomputed MSA.
Allow approximately 7 GB for downloads and 15 GB of free disk space including
extracted files, in addition to the Python environment. Downloads can be resumed
by rerunning the command.

`ATMOS_ROOT_DIR` and `PROTENIX_ROOT_DIR` default to repository `atmos_oss/` and
`af3-dev/`. Export either variable before downloading and running inference to
use a different storage location. Full training and evaluation assets are
available through the [asset download instructions](assets.md#download).

## Protenix initialization weights and common data

The common files are required for training and inference. Protenix v0.5.0
initialization weights are required for training only. Run the
following from the repository root. `PROTENIX_ROOT_DIR` defaults to repository
`af3-dev/`; set it to another directory if needed.

```bash
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-$PWD/af3-dev}"
mkdir -p "$PROTENIX_ROOT_DIR/checkpoint" "$PROTENIX_ROOT_DIR/common"

wget -c -P "$PROTENIX_ROOT_DIR/checkpoint" \
    https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v0.5.0.pt
wget -c -P "$PROTENIX_ROOT_DIR" \
    https://protenix.tos-cn-beijing.volces.com/protenix.20260101/common.tar.gz
tar -xzf "$PROTENIX_ROOT_DIR/common.tar.gz" -C "$PROTENIX_ROOT_DIR"

wget -c -P "$PROTENIX_ROOT_DIR/common" \
    https://af3-dev.tos-cn-beijing.volces.com/release_data/components.v20240608.cif
wget -c -P "$PROTENIX_ROOT_DIR/common" \
    https://af3-dev.tos-cn-beijing.volces.com/release_data/components.v20240608.cif.rdkit_mol.pkl
```

ATMOS uses the versioned CCD files `components.v20240608.cif` and
`components.v20240608.cif.rdkit_mol.pkl`; download both in addition to the common
archive. Training initializes from Protenix v0.5.0. Inference uses the four
ATMOS weights supplied in the asset package.

After setup, the common dependency paths are:

```text
af3-dev/
├── checkpoint/
│   └── protenix_base_default_v0.5.0.pt
└── common/
    ├── components.v20240608.cif
    ├── components.v20240608.cif.rdkit_mol.pkl
    ├── clusters-by-entity-40.txt
    ├── obsolete_release_date.csv
    ├── obsolete_to_successor.json
    ├── release_date_cache.json
    └── seq_to_pdb_index.json
```

Training and inference write runs under repository `output/` by default.
Set `OUTPUT_ROOT_DIR` to choose another output location.
