"""Split the mdCATH training pool into training and validation proteins."""

import argparse
import csv
import io
import random
from pathlib import Path


def read_split(path):
    """Read 320 K protein entries, rejecting duplicate protein IDs."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["name", "seqres"]:
            raise ValueError(f"{path}: expected columns name,seqres")
        rows = list(reader)

    protein_ids = set()
    for row in rows:
        name = row["name"]
        if (
            not name
            or not name.startswith("mdcath_dataset_")
            or not name.endswith("_320")
            or not row["seqres"]
            or None in row
        ):
            raise ValueError(f"{path}: invalid 320 K protein entry: {row}")
        protein_id = name.removeprefix("mdcath_dataset_").removesuffix("_320").lower()
        if not protein_id or protein_id in protein_ids:
            raise ValueError(f"{path}: empty or duplicate protein ID: {name}")
        protein_ids.add(protein_id)
    if not rows:
        raise ValueError(f"{path}: no protein entries")
    return rows, protein_ids


def split_mdcath(train_csv, test_csv, output_dir, val_size=50, seed=42):
    """Write disjoint train/val/test CSVs, preserving source row order and case."""
    pool, pool_ids = read_split(train_csv)
    test, test_ids = read_split(test_csv)
    overlap = pool_ids & test_ids
    if overlap:
        raise ValueError(f"Training pool overlaps test proteins: {sorted(overlap)}")
    if not 0 < val_size < len(pool):
        raise ValueError(f"val-size must be between 1 and {len(pool) - 1}")

    val_indices = set(random.Random(seed).sample(range(len(pool)), val_size))
    splits = {
        "train": [row for i, row in enumerate(pool) if i not in val_indices],
        "val": [row for i, row in enumerate(pool) if i in val_indices],
        "test": test,
    }
    inputs = {train_csv.resolve(), test_csv.resolve()}
    outputs = {}
    for name, rows in splits.items():
        path = output_dir / f"{name}.csv"
        if path.resolve() in inputs:
            raise ValueError(f"Output would overwrite an input CSV: {path}")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=["name", "seqres"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        contents = buffer.getvalue().encode("utf-8")
        if path.exists() and path.read_bytes() != contents:
            raise FileExistsError(
                f"{path} contains a different split; choose another --output-dir"
            )
        outputs[path] = contents

    output_dir.mkdir(parents=True, exist_ok=True)
    for path, contents in outputs.items():
        if not path.exists():
            path.write_bytes(contents)
    return {name: len(rows) for name, rows in splits.items()}


def main():
    source_dir = Path(__file__).resolve().parent / "splits"
    parser = argparse.ArgumentParser(
        description=(
            "Sample validation proteins from the 1,000-protein mdCATH training pool. "
            "All replicas of a protein belong to the same split; test proteins "
            "are copied unchanged."
        )
    )
    parser.add_argument(
        "--train-csv", type=Path, default=source_dir / "train.csv",
        help="Training pool with name,seqres columns (default: bundled splits/train.csv).",
    )
    parser.add_argument(
        "--test-csv", type=Path, default=source_dir / "test.csv",
        help="Independent test proteins (default: bundled splits/test.csv).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for train.csv, val.csv, and test.csv; must differ from the source directory.",
    )
    parser.add_argument("--val-size", type=int, default=50, help="Validation proteins (default: 50).")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42).")
    args = parser.parse_args()
    try:
        counts = split_mdcath(
            args.train_csv, args.test_csv, args.output_dir, args.val_size, args.seed
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Seed: {args.seed}; proteins: {counts}; output: {args.output_dir}")
    print("Create an H5 file list and run load_mdcath.py separately for each split.")


if __name__ == "__main__":
    main()
