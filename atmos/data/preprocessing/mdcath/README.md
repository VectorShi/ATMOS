# mdCATH data preprocessing

Prepare the mdCATH training, validation, and test data at **320 K**, using all
five simulation replicas per protein. Run the commands below from the repository
root using an environment with the dependencies in
[requirements.txt](../../../../requirements.txt); see the
[installation instructions](../../../../docs/installation.md#environment-setup).

The [prepared asset package](../../../../docs/assets.md) includes all three
splits and is ready for use with the training and inference scripts.
Use the [ModelScope download instructions](../../../../docs/assets.md#download)
to obtain the processed data and MSA.
The commands below describe preparation from raw H5 data.

## Data splits

The bundled [splits/train.csv](splits/train.csv) contains a pool of 1,000
training/validation proteins. [split_mdcath.py](split_mdcath.py) samples 50
validation proteins without replacement and assigns the remaining 950 to
training. The independent 64-protein [test list](splits/test.csv) is preserved.

| Generated split file | Proteins | Replicas per protein | Preprocessing dataset name / configuration key |
| --- | ---: | ---: | --- |
| `train.csv` | 950 | 5 | `mdCATH_tempo_train` |
| `val.csv` | 50 | 5 | `mdCATH_tempo_valid` |
| `test.csv` | 64 | 5 | `mdCATH_tempo_test` |

The split operates on proteins, so all replicas of a protein stay in one set.
The script checks for duplicate protein IDs and overlap with the test set.
Input and output CSVs have two columns:

- `name`: the mdCATH identifier followed by the temperature, for example
  `mdcath_dataset_1cmwA01_320`.
- `seqres`: the protein sequence associated with the split entry.

Remove the `_320` suffix and append `.h5` to obtain the raw filename:
`mdcath_dataset_1cmwA01.h5`. Preserve the capitalization of the identifier when
locating or downloading H5 files.

## Data locations

Keep the raw H5 files on the dataset volume and the processed data under
`$ATMOS_ROOT_DIR/mdCATH_tempo`. For example:

```bash
export MDCATH_RAW_ROOT=/path/to/raw-mdCATH
export ATMOS_ROOT_DIR=/path/to/atmos-data
export PROTENIX_ROOT_DIR=/path/to/af3-dev
export MDCATH_PREP_ROOT="$ATMOS_ROOT_DIR/mdCATH_tempo"
```

On another machine, change these roots to the corresponding storage locations.
`MDCATH_RAW_ROOT` is the parent of `data/`; Hugging Face preserves this
subdirectory when downloading. Existing data can be used in place, without
copying H5 files into the code repository.

```text
$MDCATH_RAW_ROOT/
└── data/
    └── mdcath_dataset_<protein>.h5

$MDCATH_PREP_ROOT/
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── file_lists/
│   ├── train_h5.txt
│   ├── val_h5.txt
│   └── test_h5.txt
├── mdCATH_tempo_train/
│   └── mdcath_dataset_<protein>_<replica>.pkl.gz
├── mdCATH_tempo_train.csv
├── mdCATH_tempo_valid/
│   └── mdcath_dataset_<protein>_<replica>.pkl.gz
├── mdCATH_tempo_valid.csv
├── mdCATH_tempo_test/
│   └── mdcath_dataset_<protein>_<replica>.pkl.gz
└── mdCATH_tempo_test.csv
```

The output directory and index names above are the names expected by
[configs/configs_data.py](../../../../configs/configs_data.py). The CSVs under
`splits/` list proteins; the CSVs beside the trajectory directories are processed
chain/interface indices consumed by training and inference.

## Create the training and validation splits

```bash
python atmos/data/preprocessing/mdcath/split_mdcath.py \
  --output-dir "$MDCATH_PREP_ROOT/splits" \
  --val-size 50 \
  --seed 42
```

The seed and input CSV order determine validation membership. Output rows retain
their input order and identifier capitalization. Keep the generated CSVs with
the training run for reproducibility. Repeating this command with identical
inputs and settings reuses the same output files; a different split requires a
different output directory. The bundled 1,000-protein pool is not overwritten.

## Download the H5 files

The raw files are available in the official
[compsciencelab/mdCATH dataset](https://huggingface.co/datasets/compsciencelab/mdCATH/tree/main/data)
on Hugging Face. Each H5 file contains one protein's simulations across
temperatures and replicas; the preprocessing command selects 320 K after
download.

The following example downloads the files named in the three generated CSVs
and reuses H5 files already present under `$MDCATH_RAW_ROOT/data`. If all files
are already available, continue to the H5 file lists section.

```bash
python -m pip install huggingface_hub

python - <<'PY'
import csv
import os
from pathlib import Path

from huggingface_hub import snapshot_download

split_dir = Path(os.environ["MDCATH_PREP_ROOT"]) / "splits"
raw_root = Path(os.environ["MDCATH_RAW_ROOT"])
filenames = []
for split in ("train", "val", "test"):
    with (split_dir / f"{split}.csv").open(newline="") as handle:
        filenames.extend(
            f"data/{row['name'].removesuffix('_320')}.h5"
            for row in csv.DictReader(handle)
        )

missing = sorted({name for name in filenames if not (raw_root / name).is_file()})
if missing:
    snapshot_download(
        repo_id="compsciencelab/mdCATH",
        repo_type="dataset",
        local_dir=raw_root,
        allow_patterns=missing,
        max_workers=4,
    )
else:
    print("All H5 files for the train, validation, and test splits are already present.")
PY
```

This uses Hugging Face's
[file filtering and local directory download](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-a-local-folder)
support. For test data only, use `("test",)` in place of `("train", "val", "test")`
in this example and in the file list command below.

## Create the H5 file lists

The preprocessing script accepts one H5 path per line. Generate the lists in
split CSV order and check that every input file exists:

```bash
python - <<'PY'
import csv
import os
from pathlib import Path

split_dir = Path(os.environ["MDCATH_PREP_ROOT"]) / "splits"
raw_dir = Path(os.environ["MDCATH_RAW_ROOT"]).resolve() / "data"
list_dir = Path(os.environ["MDCATH_PREP_ROOT"]) / "file_lists"
list_dir.mkdir(parents=True, exist_ok=True)

for split in ("train", "val", "test"):
    with (split_dir / f"{split}.csv").open(newline="") as handle:
        paths = [
            raw_dir / f"{row['name'].removesuffix('_320')}.h5"
            for row in csv.DictReader(handle)
        ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing H5 files:\n" + "\n".join(missing))
    output = list_dir / f"{split}_h5.txt"
    output.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
    print(f"{split}: {len(paths)} H5 paths written to {output}")
PY
```

## Run preprocessing

Use [load_mdcath.py](load_mdcath.py) to convert the H5 trajectories into
trajectory pickles and CSV indices. The script removes hydrogens and waters,
processes each frame through the Protenix structure pipeline, and stores all
frames in `atom_array_stack`. Each replica produces one `.pkl.gz` file; filenames
use lowercase protein IDs and a replica suffix from `_0` to `_4`.

Install the project's Python dependencies from
[requirements.txt](../../../../requirements.txt). The data configuration also
requires these CCD assets:

```text
$PROTENIX_ROOT_DIR/common/components.v20240608.cif
$PROTENIX_ROOT_DIR/common/components.v20240608.cif.rdkit_mol.pkl
```

Run `load_mdcath.py` **once per split**, using the corresponding file list and
dataset name. Do not pass the full 1,000-protein pool as the training list.

Process the training split (950 proteins):

```bash
python atmos/data/preprocessing/mdcath/load_mdcath.py \
  --dataset_name mdCATH_tempo_train \
  --fpath_list "$MDCATH_PREP_ROOT/file_lists/train_h5.txt" \
  --temperature 320 \
  --save_dir "$MDCATH_PREP_ROOT" \
  --tmp_root "$MDCATH_PREP_ROOT/tmp_train" \
  --max_workers 16 \
  --num_replicas 5
```

Process the validation split (50 proteins):

```bash
python atmos/data/preprocessing/mdcath/load_mdcath.py \
  --dataset_name mdCATH_tempo_valid \
  --fpath_list "$MDCATH_PREP_ROOT/file_lists/val_h5.txt" \
  --temperature 320 \
  --save_dir "$MDCATH_PREP_ROOT" \
  --tmp_root "$MDCATH_PREP_ROOT/tmp_valid" \
  --max_workers 16 \
  --num_replicas 5
```

Process the test split (64 proteins):

```bash
python atmos/data/preprocessing/mdcath/load_mdcath.py \
  --dataset_name mdCATH_tempo_test \
  --fpath_list "$MDCATH_PREP_ROOT/file_lists/test_h5.txt" \
  --temperature 320 \
  --save_dir "$MDCATH_PREP_ROOT" \
  --tmp_root "$MDCATH_PREP_ROOT/tmp_test" \
  --max_workers 16 \
  --num_replicas 5
```

Adjust `--max_workers` to the available CPU and memory. Complete preprocessing
produces 4,750 training pickles, 250 validation pickles, and 320 test pickles.
CSV indices describe chains and interfaces, so their row counts can exceed the
number of trajectory files. Both the 20 ns and 1 ns training configurations use
these same pickles and sample their frame intervals when loading data.

If the output CSV already exists, `load_mdcath.py` exits without processing that
split. Check the processing log for failed H5 files and the output file counts
before treating the dataset as complete; CSV existence alone does not establish
completeness. Before reprocessing a changed split, move its existing output CSV
and trajectory directory aside so that the loader can produce a new dataset.
The protein IDs in each processed index must match its generated split CSV.

## Training and validation

The training scripts use `mdCATH_tempo_train` for optimization and
`mdCATH_tempo_valid` for periodic validation. The trainer's `data.test_sets`
option specifies evaluation datasets and is set to `mdCATH_tempo_valid` in both
scripts. `mdCATH_tempo_test` is used by the forward-simulation script.

All three dataset configurations enable protein MSA and share these assets:

```text
$MDCATH_PREP_ROOT/mdcath_msa/seq_to_pdb_index.json
$MDCATH_PREP_ROOT/mdcath_msa/msa_template/
```

Provide the sequence-to-MSA index and MSA files for the proteins in each split;
`load_mdcath.py` prepares structures and trajectories, not MSAs. See
[training instructions](../../../../README.md#training-and-data-processing) for model initialization,
training settings, and launch commands.
