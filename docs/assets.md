# Prepared data and model assets

The asset package provides trajectory pickles, chain/interface indices,
protein MSA files and sequence mappings, fixed data splits, four ATMOS model
checkpoints with YAML configurations, and mdCATH evaluation references.
Download Protenix initialization weights and common CCD data separately using
the [installation instructions](installation.md#protenix-initialization-weights-and-common-data). Keep the
directory structure below when downloading or extracting the assets.

```text
atmos_oss/
├── packages.json
├── msa_coverage.json
├── atmos_checkpoints/
│   ├── mdCATH_i20.pt
│   ├── mdCATH_i20_config.yaml
│   ├── mdCATH_i1.pt
│   ├── mdCATH_i1_config.yaml
│   ├── misato_i10.pt
│   ├── misato_i10_config.yaml
│   ├── misato_i1.pt
│   └── misato_i1_config.yaml
├── mdCATH_tempo/
│   ├── splits/                         # train.csv, val.csv, test.csv
│   ├── mdCATH_tempo_train.csv
│   ├── mdCATH_tempo_train/
│   ├── mdCATH_tempo_valid.csv
│   ├── mdCATH_tempo_valid/
│   ├── mdCATH_tempo_test.csv
│   ├── mdCATH_tempo_test/
│   ├── mdcath_msa/
│   │   ├── seq_to_pdb_index.json
│   │   └── msa_template/
│   ├── mdCATH_tempo_test_gt_trajectories/
│   └── mdCATH_tempo_test_ref_pdb/
└── misato/
    ├── splits/                         # train_MD.txt, val_MD.txt, test_MD.txt
    ├── train_MD.csv
    ├── train_MD/
    ├── val_MD.csv
    ├── val_MD/
    ├── test_MD.csv
    ├── test_MD/
    ├── seq_to_pdb_index.json
    └── msa_template/
```

The four `.pt` files provide ATMOS inference weights. Each model uses the YAML
configuration with the matching name prefix. The `i20`, `i10`, and `i1` names
identify the trajectory sampling interval: mdCATH uses 20 ns and 1 ns; MISATO
uses 10 stored frames and 1 frame.

The CSVs beside trajectory directories are loader indices, with one row per
chain or interface. A trajectory may have multiple index rows. Files under
`splits/` specify protein or complex membership. MSA directories contain
`pairing.a3m` and `non_pairing.a3m`; the JSON mappings connect protein
sequences to those directory names. Templates are disabled for these tasks.
`msa_coverage.json` lists chains whose protein sequences have no exact match
in the supplied MSA index. Those chains use query-only features. This affects
7 mdCATH proteins (6 train, 1 test) and 14 MISATO complexes (9 train,
5 validation); all 100 MISATO test complexes have indexed protein sequences.

| Dataset | Train | Validation | Test | Trajectories |
| --- | ---: | ---: | ---: | --- |
| mdCATH | 950 proteins | 50 proteins | 64 proteins | Five replicas per protein at 320 K; 4,750 / 250 / 320 pickles |
| MISATO | 12,662 complexes | 1,258 complexes | 100 complexes | One 100-frame pickle per complex |

All replicas of a protein belong to the same split. MISATO preserves the
dataset's train/validation/test membership, applies preprocessing exclusions,
and uses the fixed 100-complex test subset. Training pickles contain
`atom_array_stack`, plus the first frame's atom/token arrays and entity and
sequence metadata. The dataset code samples trajectory windows at load time;
the coarse and fine models use the same files.

## Licenses and attribution

The repository's Apache 2.0 license covers source code, subject to the
[third-party notices](../THIRD_PARTY_NOTICES.md). It does not license the
entire asset package.

The source trajectory datasets have the following attribution and licenses:

| Source | Attribution | License |
| --- | --- | --- |
| [mdCATH](https://huggingface.co/datasets/compsciencelab/mdCATH) | Mirarchi, Giorgino, and De Fabritiis; [dataset paper](https://doi.org/10.1038/s41597-024-04140-z) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [MISATO](https://zenodo.org/records/7711953) | Siebenmorgen and colleagues; full creator list in the [dataset record](https://doi.org/10.5281/zenodo.7711953) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

ATMOS distributes processed subsets of these trajectories, with converted
structure and trajectory formats, filtering, data splits, and loader indices
as described above. Preserve the source attribution when using or
redistributing these data.

These trajectory-data licenses do not establish the license of model
checkpoints, MSA sequence databases, CCD files, or other inputs. Use the terms
provided with each respective distribution. Protenix initialization uses the
v0.5.0 checkpoint linked in the installation instructions. PDBbind source
files are obtained separately under the
[access and license terms](../atmos/data/preprocessing/misato/README.md#pdbbind-v2020-inputs-and-access)
described in the preprocessing guide.

## Download

ATMOS models and prepared datasets are available on ModelScope:

| Repository | Contents |
| --- | --- |
| [SmartVector/ATMOS](https://modelscope.cn/models/SmartVector/ATMOS) | Four inference weight files and their YAML configurations |
| [SmartVector/ATMOS-dataset](https://modelscope.cn/datasets/SmartVector/ATMOS-dataset) | Prepared mdCATH and MISATO trajectories, indices, MSA, fixed splits, and mdCATH evaluation references |

From the code repository root, install the
[ModelScope Hub client](https://github.com/modelscope/modelscope_hub) in the
project environment and download both repositories into the same asset root.
The commands below default to repository `atmos_oss/`; set `ATMOS_ROOT_DIR`
to another storage location before running them if needed.

```bash
python -m pip install 'modelscope-hub>=0.5.0'
export ATMOS_ROOT_DIR="${ATMOS_ROOT_DIR:-$PWD/atmos_oss}"
mkdir -p "$ATMOS_ROOT_DIR"

ms-hub --endpoint https://modelscope.cn download SmartVector/ATMOS \
    --repo-type model --include 'atmos_checkpoints/*' \
    --local-dir "$ATMOS_ROOT_DIR"
ms-hub --endpoint https://modelscope.cn download SmartVector/ATMOS-dataset \
    --repo-type dataset \
    --include 'mdCATH_tempo/*' 'misato/*' packages.json msa_coverage.json \
    --local-dir "$ATMOS_ROOT_DIR"
```

After both downloads complete, extract the data archives using Bash and GNU tar:

```bash
find "$ATMOS_ROOT_DIR/mdCATH_tempo" "$ATMOS_ROOT_DIR/misato" \
    -type f \( -name '*.tar' -o -name '*.tar.gz' \) -print0 |
while IFS= read -r -d '' archive; do
    tar -xf "$archive" -C "$ATMOS_ROOT_DIR" || exit 1
done
```

Every archive, including MSA archives, contains paths relative to
`ATMOS_ROOT_DIR`. Extract all numbered parts into this root; each part is an
independent archive. CSV indices, split lists, and JSON mappings retain their
downloaded paths. `packages.json` lists the archives and accompanying files.
The extracted directory layout and dataset details are described above.

Inference weights and YAMLs are read from
`$ATMOS_ROOT_DIR/atmos_checkpoints/`. The four weight files are
`mdCATH_i20.pt`, `mdCATH_i1.pt`, `misato_i10.pt`, and `misato_i1.pt`.
Each file provides inference weights for its corresponding model and is paired
with a YAML configuration. The prepared datasets are ready for training and
inference.

## Using the downloaded package

Download weights and their configurations from the ModelScope model repository
[SmartVector/ATMOS](https://modelscope.cn/models/SmartVector/ATMOS), and data
archives and indices from the dataset repository
[SmartVector/ATMOS-dataset](https://modelscope.cn/datasets/SmartVector/ATMOS-dataset).
Follow the [download and extraction commands](#download)
to place both repositories' files under the same `ATMOS_ROOT_DIR`.
The dataset's `packages.json` lists the archives, their sizes and contents,
metadata paths, and model filenames.

The default asset root is `atmos_oss/` in the ATMOS code repository. Place the
package there, create a directory link to its storage location, or set
`ATMOS_ROOT_DIR` explicitly. This directory is excluded from Git.
After extracting the download, this root must directly contain
`mdCATH_tempo/`, `misato/`, and `atmos_checkpoints/`, as shown above.
Every archive, including MSA archives, stores paths relative to this root.
Extract each archive into `ATMOS_ROOT_DIR`; the stored paths create the required
subdirectories. Each numbered archive is independently extractable. Download
and extract all parts for each required dataset and split.

CSV indices, split lists, JSON mappings, and model files retain their downloaded
relative paths. Model weights and YAMLs are used directly without extraction.
Use the prepared files directly with the training and inference commands below.

Obtain the ATMOS code and enter its repository:

```bash
git clone https://github.com/VectorShi/ATMOS.git
cd ATMOS
```

Install the [project environment](installation.md#environment-setup), including
the dependencies in [requirements.txt](../requirements.txt). CUDA GPUs and
the configured attention kernels are required for the supplied GPU launch
commands. The default kernels are `triattention` and `cuequivariance`; set
`TRIANGLE_ATTENTION=torch TRIANGLE_MULTIPLICATIVE=torch` to use the PyTorch
implementations. Pickles contain Biotite and Protenix objects and must be read
with the repository and its compatible Python dependencies available.

Download the [common dependencies](installation.md#protenix-initialization-weights-and-common-data),
then run the following from the ATMOS code repository in the installed
environment, replacing the paths:

```bash
export ATMOS_ROOT_DIR=/path/to/atmos_oss
export PROTENIX_ROOT_DIR=/path/to/af3-dev
export CHECKPOINT_DIR="$ATMOS_ROOT_DIR/atmos_checkpoints"
export CONFIG_DIR="$CHECKPOINT_DIR"
export OUTPUT_ROOT_DIR=/path/to/atmos-output
```

`ATMOS_ROOT_DIR` supplies the datasets and ATMOS weights.
`PROTENIX_ROOT_DIR` supplies `common/` and `checkpoint/` and defaults to
repository `af3-dev/`. `CONFIG_DIR` is used by the MISATO
inference launcher and defaults to `CHECKPOINT_DIR`. mdCATH reads its YAMLs
from `CHECKPOINT_DIR`.
The YAMLs specify model and sampling parameters; dataset paths come from the
code configuration and launch environment.
Training writes to repository `output/` by default. Inference and parallel
launch logs use `output/mdcath_inference/` or `output/misato_inference/`.
`OUTPUT_ROOT_DIR` overrides these destinations.

Inspect both inference commands after configuring the paths:

```bash
bash atmos/run_scripts/mdcath_forward_simulation.sh --dry-run
bash atmos/run_scripts/misato_forward_simulation.sh --msa --dry-run
```

Use the training, inference, and evaluation commands below for the desired
task. Launch each training or inference command as a separate job.

## Training

Run each training command as a separate job from the code repository:

```bash
GPUS_PER_NODE=8 bash atmos/run_scripts/finetune_mdcath_20ns.sh
GPUS_PER_NODE=8 bash atmos/run_scripts/finetune_mdcath_1ns_interp.sh
GPUS_PER_NODE=8 bash atmos/run_scripts/finetune_misato_i10.sh
GPUS_PER_NODE=8 bash atmos/run_scripts/finetune_misato_i1.sh
```

Training initializes from
`$PROTENIX_ROOT_DIR/checkpoint/protenix_base_default_v0.5.0.pt` and
freezes the static embedding modules and 48-layer Pairformer. It uses protein
MSA and `N_cycle=4`: training samples one to four recycling passes, and
validation uses four. Train and validation sets are separate; test
targets are reserved for forward simulation. `INIT_CHECKPOINT_PATH` overrides
the initialization weight path.

See the [mdCATH training guide](../atmos/data/preprocessing/mdcath/README.md#training-and-validation)
and [MISATO training guide](../atmos/data/preprocessing/misato/README.md#training)
for sampling intervals, model history lengths, and optimization settings.

## Inference

These commands distribute targets across eight GPUs. Set `GPUS_PER_NODE=1`
for one GPU:

```bash
GPUS_PER_NODE=8 bash atmos/run_scripts/mdcath_forward_simulation.sh
GPUS_PER_NODE=8 bash atmos/run_scripts/misato_forward_simulation.sh --msa
```

Each launch prints its log location and creates a timestamped run directory
under `output/mdcath_inference/` or `output/misato_inference/` (or
`OUTPUT_ROOT_DIR` when set). The run directory contains `predictions/`; use
that run directory as `--work-dir` for evaluation.

| Dataset | Coarse / fine checkpoints | History windows | Trajectory schedule | Maximum MSA rows per recycling pass |
| --- | --- | --- | --- | ---: |
| mdCATH | `mdCATH_i20.pt` / `mdCATH_i1.pt` | 20 / 30 frames | `20:20;20:1` | 16,384 |
| MISATO | `misato_i10.pt` / `misato_i1.pt` | 10 / 20 frames | `10:10;10:1` | 16,384 |

Both use MSA, `N_cycle=4`, stitching prefix 5, 50 diffusion steps, one sample,
noise scale `1.5`, and step scale `1.5`, with seed 42, BF16, and a 48-layer
Pairformer. The fine model uses destination
conditioning and requires unit transition intervals; the first interval is the
destination horizon. Add `--dry-run` to inspect a command. The parallel-halves
launchers run two processes per GPU using the same model and data settings:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash atmos/run_scripts/mdcath_forward_simulation_parallel_halves.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash atmos/run_scripts/misato_forward_simulation_parallel_halves.sh --msa
```

## Evaluation

For mdCATH evaluation, use the GT trajectories and crystal references included
in the package:

```bash
bash atmos/evaluation/mdcath/run_mdcath_evaluation.sh \
    --work-dir /path/to/mdcath-inference-run \
    --gt-dir "$ATMOS_ROOT_DIR/mdCATH_tempo/mdCATH_tempo_test_gt_trajectories" \
    --ref-dir "$ATMOS_ROOT_DIR/mdCATH_tempo/mdCATH_tempo_test_ref_pdb"
```

For a parallel mdCATH run, collect both groups' prediction files in one
`predictions/` directory and pass its parent as `--work-dir`.

MISATO inference prediction pickles contain the corresponding reference
trajectory for evaluation. Its evaluator also accepts the parallel launch
directory directly:

```bash
bash atmos/evaluation/misato/run_misato_evaluation.sh \
    --work-dir /path/to/misato-inference-run
```

See the [mdCATH](../atmos/evaluation/mdcath/README.md) and
[MISATO](../atmos/evaluation/misato/README.md) evaluation guides for metric
definitions, supported input formats, and output files.
