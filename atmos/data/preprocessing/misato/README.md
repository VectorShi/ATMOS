# MISATO data preprocessing

For training and inference with prepared data, follow the
[ModelScope download instructions](../../../../docs/assets.md#download)
to obtain the trajectories, indices, MSA, weights, and configurations. The
commands below describe preparation from raw data.

Run the commands from the repository root using an environment containing the
dependencies for the corresponding step. Step 0 requires `pytraj`, `h5py`,
`numpy`, and `tqdm`; the conversion has been tested with Python 3.7 and
`pytraj` 2.0.5. Subsequent steps use the project's preprocessing dependencies.

## Data splits and sizes

Steps 0–4 process trajectories and record failures. Step 5 produces the train,
validation, and fixed-test-subset indices. Counts below refer to unique PDB IDs,
not interface-index rows. Each complex has a 100-frame trajectory.

| Split | Input complexes | Complexes passing preprocessing | Complexes used |
| --- | ---: | ---: | ---: |
| Train | 13,765 | 12,662 | 12,662 |
| Validation | 1,595 | 1,258 | 1,258 |
| Test | 1,612 | 1,276 | 100 |

The three input splits are disjoint. The
[exclusion list](excluded_pdb_list.txt) contains 1,776 complexes that fail
preprocessing. The final test set is the fixed subset in
[misato_test_include_only_list.txt](misato_test_include_only_list.txt); all
100 IDs belong to the test split and pass preprocessing. Step 5 applies this
list to the eligible test indices. The remaining 1,176
eligible test complexes are not reassigned to training or validation.

## PDBbind v2020 inputs and access

Ligand naming uses **PDBbind v2020**. Download the following v2020 packages
yourself from the official [PDBbind download page](https://www.pdbbind-plus.org.cn/download),
using your own account and accepting the applicable license terms:

| Package | Archive | Use |
| --- | --- | --- |
| Protein–ligand refined set | `PDBbind_v2020_refined.tar.gz` | Primary MOL2 files for 5,316 complexes. |
| Protein–ligand general set minus refined set | `PDBbind_v2020_other_PL.tar.gz` | Primary MOL2 files for 14,127 complexes. |
| Ligand molecules in the general set, MOL2 format | `PDBbind_v2020_mol2.tar.gz` | Fallback ligand MOL2 files. |

The refined and general-minus-refined sets together cover the 19,443-complex
v2020 general set. Use the v2020 files named above to reproduce the ligand
names and atom ordering used by this preprocessing workflow.

**PDBbind data are not redistributed with this repository.** Keep downloaded
archives and extracted PDBbind files outside the repository, and do not include
them in public code or data releases. Obtain the files directly from the data
provider and follow its [license terms](https://www.pdbbind-plus.org.cn/termofuse),
including restrictions on redistribution.

Extract the archives locally. Combine the per-complex directories from the
refined and general-minus-refined packages under the primary MOL2 directory;
place the general-set ligand MOL2 files in the fallback directory:

```text
$MISATO_ROOT/PDBBind_v2020/
├── complex_ligands/
│   └── <lowercase_id>/
│       └── <lowercase_id>_ligand.mol2
└── mol2/
    └── <lowercase_id>_ligand.mol2
```

`$MISATO_ROOT` is your local dataset root, as set below. These directory names
match the step 2 command example; use `--pdbbindMol2` and `--pdbbindMol2_alt` to
point to your own extracted locations.

## Step 0: H5 coordinates to trajectories

Prepare the following files. Replace `/path/to/MISATO` with the dataset location
on your machine. Keep raw data and generated trajectories outside the code
repository.

```text
/path/to/MISATO/
├── MD.hdf5
├── parameter_restart_files_MD.tar.gz
├── parameter_restart_files_MD/
│   └── <lowercase_pdb_id>/
│       ├── production.rst
│       └── production.top.gz
└── splits/
    ├── train_MD.txt
    ├── val_MD.txt
    └── test_MD.txt
```

The split files contain one uppercase PDB ID per line: 13,765 training, 1,595
validation, and 1,612 test complexes. Step 0 processes all 16,972 IDs and does not
apply an exclusion list.

Extract the restart archive if these inputs are not already available:

```bash
export MISATO_ROOT=/path/to/MISATO
tar -xzf "$MISATO_ROOT/parameter_restart_files_MD.tar.gz" \
    -C "$MISATO_ROOT" --skip-old-files
```

For each complex, the converter reads `production.rst` and `production.top`.
When only `production.top.gz` is present, it decompresses the topology beside
the compressed file, so the restart directory must be writable.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python atmos/data/preprocessing/misato/step_0_misato_h5_to_traj.py \
    -s "$MISATO_ROOT/splits" \
    -dMD "$MISATO_ROOT/MD.hdf5" \
    -r "$MISATO_ROOT/parameter_restart_files_MD" \
    -o "$MISATO_ROOT/processed/step0" \
    -n 24
```

The conversion removes residues named `WAT`, `Na+`, and `Cl-` from the restart
topology, then uses the H5 coordinates for every trajectory frame. Hydrogens are
retained at this step. The H5 atom count must match the stripped topology.

The output directory contains:

- `<PDB_ID>.nc` and `<PDB_ID>.top`: NetCDF coordinates and the matching AMBER topology.
- `successes.jsonl`: completed IDs with frame and atom counts.
- `failed_pdb_ids.txt` and `failures.jsonl`: failed IDs and their error details.
- `summary.json`: progress, success/failure counts, worker count, and run status.

Individual input errors are recorded while other complexes continue processing.
Use a separate output directory for each run to keep its results and failure
records together. Review `summary.json` and the failure records before running
subsequent steps.

## Step 1: chain assignment and hydrogen removal

Use an environment containing `mdtraj`, `h5py`, `numpy`, `pandas`, and `tqdm`.
Place the MISATO mapping files `atoms_residue_map.pickle`,
`atoms_type_map.pickle`, and `atoms_name_map_for_pdb.pickle` in
`$MISATO_ROOT/maps/`.

After step 0 completes, add its failed IDs to `excluded_pdb_list.txt`. If there
are no failures, the list remains empty.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python atmos/data/preprocessing/misato/step_1_split_chain.py \
    -s "$MISATO_ROOT/splits" \
    -e atmos/data/preprocessing/misato/excluded_pdb_list.txt \
    -dMD "$MISATO_ROOT/MD.hdf5" \
    -t "$MISATO_ROOT/processed/step0" \
    -m "$MISATO_ROOT/maps" \
    -o "$MISATO_ROOT/processed/step1" \
    -n 24
```

The converter reads AMBER `.top` or `.prmtop` files, assigns chains from the H5
metadata, removes hydrogen atoms, and writes `<PDB_ID>.xtc` with all frames and
`<PDB_ID>.pdb` with the first frame. Input topology files remain in place.
`summary.json` tracks progress; `successes.jsonl`, `failed_pdb_ids.txt`, and
`failures.jsonl` record completed IDs and errors. After this step completes,
append its actual failed IDs to the exclusion list before continuing.

## Step 2: ligand residue and atom names

Use the step 1 PDB/XTC trajectories and the PDBbind v2020 MOL2 files described
above. The primary MOL2
directory contains `<lowercase_id>/<lowercase_id>_ligand.mol2`; the fallback
directory contains `<lowercase_id>_ligand.mol2` directly. The fallback is used
only when the primary file is absent. Set these paths to your MOL2 locations.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python atmos/data/preprocessing/misato/step_2_add_ccd.py \
    -s "$MISATO_ROOT/splits" \
    -e atmos/data/preprocessing/misato/excluded_pdb_list.txt \
    -t "$MISATO_ROOT/processed/step1" \
    -p "$MISATO_ROOT/PDBBind_v2020/complex_ligands" \
    -p1 "$MISATO_ROOT/PDBBind_v2020/mol2" \
    -o "$MISATO_ROOT/processed/step2" \
    -n 24
```

The input must have one `MOL` residue at the end of the topology, beginning its
own chain. Its heavy-atom element sequence must match the MOL2 atom order.
Ligand residue and atom names are then assigned from the MOL2 file while
preserving the trajectory coordinates.

Missing MOL2 files, incompatible `MOL` placement, and mismatched element
sequences are recorded as individual failures. Other complexes continue
processing. `summary.json` reports progress, `successes.jsonl` lists completed
IDs, and `failed_pdb_ids.txt` and `failures.jsonl` record failures and reasons.
After this step completes, append its actual failed IDs to the exclusion list.

## Step 3: bioassembly trajectories and interface indices

Use an environment containing the project's preprocessing dependencies and
prepare the Protenix common data described in the
[installation instructions](../../../../docs/installation.md#protenix-initialization-weights-and-common-data).
Set `PROTENIX_ROOT_DIR` to the directory containing `common/` and
`ATMOS_ROOT_DIR` to your ATMOS data directory. The CCD filenames are specified
by `ccd_components_file` and `ccd_components_rdkit_mol_file` in
[configs/configs_data.py](../../../../configs/configs_data.py).

This step takes a single file of PDB IDs. Combine the three split files and
apply the exclusions accumulated through step 2:

```bash
export PROTENIX_ROOT_DIR=/path/to/af3-dev
export ATMOS_ROOT_DIR=/path/to/atmos-data
mkdir -p "$MISATO_ROOT/processed"
cat "$MISATO_ROOT/splits/train_MD.txt" \
    "$MISATO_ROOT/splits/val_MD.txt" \
    "$MISATO_ROOT/splits/test_MD.txt" > "$MISATO_ROOT/processed/all_MD.txt"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python atmos/data/preprocessing/misato/step_3_transform_to_biodict.py \
    -s "$MISATO_ROOT/processed/all_MD.txt" \
    -e atmos/data/preprocessing/misato/excluded_pdb_list.txt \
    -t "$MISATO_ROOT/processed/step2" \
    -o "$MISATO_ROOT/processed/step3" \
    -n 24
```

The converter processes every trajectory frame through the Protenix
`Distillation` data pipeline. Each `<lowercase_id>.pkl.gz` contains the first
frame's metadata, token array, and all frames in `atom_array_stack`. Atom
identities must remain consistent across frames. The first frame supplies
the interface indices containing the last chain in the parsed atom array.
A complex must have at least one such interface. This selection assumes that
the ligand remains the last chain after parsing.

`step3.csv`, beside the output directory, contains the combined interface
indices. Inside the output directory, `successes.jsonl` records completed IDs
and their interface indices as each complex finishes. `summary.json` reports
progress and `failed_pdb_ids.txt` and `failures.jsonl` record failures. Pickles
are made available only after all frames of a complex finish processing.
After the step completes, append its actual failed IDs to the exclusion list.

## Step 4: protein–ligand proximity

Read the step 1 PDB/XTC files, where ligand residues are named `MOL`, and
apply the exclusions accumulated through step 3. Use an environment containing
`mdtraj`, `torch`, and `tqdm`.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python atmos/data/preprocessing/misato/step_4_remove_unbind_ligand.py \
    -s "$MISATO_ROOT/splits" \
    -e atmos/data/preprocessing/misato/excluded_pdb_list.txt \
    -t "$MISATO_ROOT/processed/step1" \
    -o "$MISATO_ROOT/processed/step4" \
    -n 48
```

For each frame, the script computes the minimum distance between atoms named
`CA` and atoms in `MOL` residues, without periodic wrapping. All 100 frames must
have a minimum distance strictly below **12 Å**. Inputs with a different frame
count, empty atom selections, or nonfinite coordinates are recorded as errors.

`distances.jsonl` records the 100 per-frame distances and the largest of these
minimum distances for each successfully measured complex. `passed_pdb_ids.txt`
lists complexes that meet the criterion. `failures.jsonl` and
`failed_pdb_ids.txt` record distance rejections and processing errors;
`summary.json` reports progress and counts these two failure categories
separately. Records are written as each complex finishes processing.

After the step completes, append its actual failed IDs to the exclusion list.

## Step 5: split and filter interface indices

Use Python 3; this step requires only the standard library. Supply the combined
step 3 CSV, original split files, cumulative exclusions through step 4, and the
step 4 pass list:

```bash
python atmos/data/preprocessing/misato/step_5_filter_indices.py \
    -i "$MISATO_ROOT/processed/step3.csv" \
    -s "$MISATO_ROOT/splits" \
    -e atmos/data/preprocessing/misato/excluded_pdb_list.txt \
    -p "$MISATO_ROOT/processed/step4/passed_pdb_ids.txt" \
    --testIncludeFile atmos/data/preprocessing/misato/misato_test_include_only_list.txt \
    -o "$MISATO_ROOT/processed/indices"
```

The output directory contains:

| File | Complexes | Contents |
| --- | ---: | --- |
| `train_MD.csv` | 12,662 | All passing training complexes. |
| `val_MD.csv` | 1,258 | All passing validation complexes. |
| `test_MD.csv` | 100 | Passing test complexes in the fixed inclusion list. |
| `summary.json` | — | Input, excluded, passing, and output counts, including interface-row counts by split. |

Original split membership is preserved. All interface rows for retained
complexes are kept in their input order, with their field values unchanged.
The remaining passing test complexes are omitted from all three output CSVs.
Trajectory pickles remain in the step 3 output directory.

The script checks that splits do not overlap, that the pass and exclusion
lists account for every input ID without conflict, and that every passing
complex has interface indices. Every requested test ID must belong to the
original test split and pass preprocessing. Missing IDs or inconsistent
inputs cause an error before output files are written.

## Exclusions and test subset

The included `excluded_pdb_list.txt` records failures across steps 0–4. To
rebuild this list from your own inputs, start with an empty list before step 0.
After each step completes, append its actual failed IDs from
`failed_pdb_ids.txt`, removing duplicates. Supply this accumulated list to
the next step's `--excludeFile` argument.

`misato_test_include_only_list.txt` contains the fixed 100-complex test subset;
it does not restrict step 0. Step 5 applies it after the preprocessing
exclusions, matching IDs case-insensitively and keeping all eligible training
and validation IDs in their original splits.

## Training

Prepared training assets use the following layout. Each CSV contains all
ligand-interface rows for its split; each trajectory is stored once.

```text
$ATMOS_ROOT_DIR/misato/
├── train_MD.csv
├── train_MD/              # 12,662 trajectory pickles
├── val_MD.csv
├── val_MD/                # 1,258 trajectory pickles
├── test_MD.csv
├── test_MD/               # 100 trajectory pickles
├── splits/                # One complex ID per line for each split
├── seq_to_pdb_index.json
└── msa_template/
    └── <msa_id>/
        ├── pairing.a3m
        └── non_pairing.a3m
```

When preparing this layout from step 3 and step 5 outputs, copy the files named
by each final CSV into the corresponding trajectory directory. The step 3
directory also contains complexes rejected by later filtering; directory
membership alone does not define a training or evaluation split. Training
pickles contain `atom_array_stack` with 100 frames and the first frame's
`atom_array`, `token_array`, sequence, and entity metadata.

Install the [project dependencies](../../../../docs/installation.md#environment-setup),
provide the CCD common data, and place
`protenix_base_default_v0.5.0.pt` under `$PROTENIX_ROOT_DIR/checkpoint/`.
The [asset guide](../../../../docs/assets.md) describes the complete layout.
From the repository root:

```bash
export ATMOS_ROOT_DIR=/path/to/atmos-data
export PROTENIX_ROOT_DIR=/path/to/af3-dev
GPUS_PER_NODE=8 bash atmos/run_scripts/finetune_misato_i10.sh
GPUS_PER_NODE=8 bash atmos/run_scripts/finetune_misato_i1.sh
```

Launch the two training commands as separate jobs. Both use `misato_train` for
optimization and `misato_valid` for periodic validation. The fixed test set is
reserved for forward simulation and evaluation.

| Setting | `finetune_misato_i10.sh` | `finetune_misato_i1.sh` |
| --- | --- | --- |
| Frame interval | 10 stored frames | 1 stored frame |
| Maximum history length | 10 frames | 20 frames |
| Destination conditioning | Disabled | Enabled |
| Pairformer layers / recycling passes | 48 / 4 | 48 / 4 |

`N_cycle=4` is the training maximum: each step samples one to four recycling
passes, and validation uses four.

Both models initialize from Protenix v0.5.0 and freeze its embedding modules,
MSA module, and Pairformer. Protein MSA is enabled, with at most 1,024 sampled
rows per recycling pass; structural templates are disabled. Training uses a
256-token spatial interface crop with complete ligand residues, six prefixes
per batch, eight diffusion samples per prefix, BF16, learning rate `0.0005`,
2,000 warmup steps, 100,000 total steps, and EMA decay `0.999`. Validation runs
every 500 steps, and checkpoints are saved every 1,000 steps. Validation
sampling uses 50 diffusion steps, five samples, noise scale `1.003`, and step
scale `1.5`.

Set `INIT_CHECKPOINT_PATH`, `OUTPUT_ROOT_DIR`, and `GPUS_PER_NODE` to change the
initial weights, output location, and GPU count. Multi-node settings are
`NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`. Add `--dry-run` to
inspect the command without starting training.

## Inference

Use an environment installed from [requirements.txt](../../../../requirements.txt).
The entry point is [misato_forward_simulation.sh](../../../run_scripts/misato_forward_simulation.sh).
Download and extract the [ModelScope assets](../../../../docs/assets.md#download).
Its default input layout is:

```text
$ATMOS_ROOT_DIR/
├── atmos_checkpoints/
│   ├── misato_i10.pt
│   ├── misato_i10_config.yaml
│   ├── misato_i1.pt
│   └── misato_i1_config.yaml
└── misato/
    ├── test_MD.csv
    ├── test_MD/
    │   └── <lowercase_id>.pkl.gz
    ├── seq_to_pdb_index.json
    └── msa_template/
        └── <msa_id>/
            ├── pairing.a3m
            └── non_pairing.a3m
```

Set `ATMOS_ROOT_DIR` to the asset package root; it defaults to `atmos_oss/` in
the repository. `PROTENIX_ROOT_DIR` defaults to repository `af3-dev/` and must contain the
[common data](../../../../docs/installation.md#protenix-initialization-weights-and-common-data). The input index can
contain all eligible test complexes: the loader applies the fixed inclusion
list, matches its uppercase IDs to lowercase indices, and groups interface
rows by PDB ID so that each complex is inferred once.

Set `SOURCE_INDICES_CSV` and `BIOASSEMBLY_DIR` to use the step 5 index and step 3
trajectory directory directly. The loader accepts trajectory frames under
`atom_array_stack`, or a list of frames under `atom_array`. Inference starts
from the first frame and uses a spatial interface crop of at most 256 tokens,
keeping ligand residues complete. The crop is applied to every trajectory
frame with the same atom identities.

The sequence-to-MSA index maps protein sequences to the directory IDs under
`msa_template/`. Extract all MISATO MSA archive parts into `ATMOS_ROOT_DIR`
before running inference; their stored paths create `misato/msa_template/`.
`MSA_INDEX` and `MSA_DIR` can point to an index and extracted MSA directory
elsewhere.

| Setting | Coarse model | Fine model |
| --- | --- | --- |
| Checkpoint | `misato_i10.pt` | `misato_i1.pt` |
| Frame interval | 10 | 1 |
| History window | 10 frames | 20 frames |
| Destination conditioning | Disabled | Enabled |
| Pairformer layers / recycling passes | 48 / 4 | 48 / 4 |
| Diffusion steps / samples | 50 / 1 | 50 / 1 |
| Noise scale / step scale | 1.5 / 1.5 | 1.5 / 1.5 |
| Maximum sampled MSA rows per recycling pass | 16,384 | 16,384 |

The following settings apply to both stages:

| Setting | Value |
| --- | --- |
| Protein MSA / structural templates | Enabled / disabled |
| Trajectory schedule | `10:10;10:1`, producing 100 frames including the initial frame |
| Stitching prefix | 5 frames from the preceding segment |
| History overlap on window reset | 5 frames (`overlap_structure_num`) |
| Spatial interface crop | 256-token budget; complete ligand residues |
| Random seed | 42 |
| Precision | BF16 autocast; FP32 diffusion sampling |
| Temperature conditioning | Fixed model input of 320 |
| Diffusion churn | `gamma0=0.8`, `gamma_min=1.0` |
| Noise schedule | `s_max=160`, `s_min=0.0004`, `rho=7`, `sigma_data=16` |

Frame intervals are measured in stored trajectory frames. Temperature
conditioning is a fixed model input, not a temperature read from the trajectory.

Model configurations are supplied with the weights under
`$ATMOS_ROOT_DIR/atmos_checkpoints/` as `misato_i10_config.yaml` and
`misato_i1_config.yaml`; see the [asset layout](../../../../docs/assets.md).
They specify model and sampling settings; input paths are supplied by the
launch script. `CHECKPOINT_DIR` changes the weight directory, and `CONFIG_DIR`
changes the YAML directory; both default to `atmos_checkpoints/` in the asset
package. Template embedding is disabled (`n_blocks=0`). Checkpoint loading uses
`load_strict=false` to allow absent parameters for this inactive module.

The schedule `10:10;10:1` produces 100 frames using a 5-frame stitching prefix:

```bash
export ATMOS_ROOT_DIR=/path/to/atmos-data
export PROTENIX_ROOT_DIR=/path/to/af3-dev
GPUS_PER_NODE=8 bash atmos/run_scripts/misato_forward_simulation.sh
```

The script defaults to one GPU. Add `--dry-run` to inspect the command, or
`--no-msa` to use query-only sequence features without external MSA files.
Predictions and launch logs are written under repository `output/misato_inference/`;
set `OUTPUT_ROOT_DIR` to change this location.

To run two inference processes per GPU on one node, use
[misato_forward_simulation_parallel_halves.sh](../../../run_scripts/misato_forward_simulation_parallel_halves.sh):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash atmos/run_scripts/misato_forward_simulation_parallel_halves.sh
```

The launcher applies the fixed test inclusion list, splits the 100 complexes
into groups of 50 in input order, and keeps all interface rows for each complex
in its group. Both groups use GPUs 0–7 and call the same inference entry point
with the settings above. Set `CUDA_VISIBLE_DEVICES` to use another shared GPU
list, or `GPU_IDS_PART0` and `GPU_IDS_PART1` for separate lists. GPU counts are
derived from those lists. The default master ports are 29512 and 29513;
override `MASTER_PORT_PART0` and `MASTER_PORT_PART1` with distinct ports when
running additional jobs.

Both groups write under one timestamped launch directory, with separate run
directories, logs, and `indices/part0.csv` and `indices/part1.csv`. Add `--dry-run`
to write the split CSVs and inspect both commands without starting inference.
`--no-msa` applies to both groups.
