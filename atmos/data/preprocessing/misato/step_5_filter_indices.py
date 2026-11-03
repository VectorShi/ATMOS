"""Write MISATO train, validation, and fixed-test-subset interface indices.

Retain complexes passing steps 0-4 in their original splits and restrict the
test split to the supplied inclusion list. Interface fields and trajectory
files are preserved.
"""

import argparse
import csv
import json
import tempfile
from pathlib import Path


def read_ids(path):
    """Read one case-insensitive PDB ID per line, rejecting duplicates."""
    ids = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        pdb_id = line.strip().upper()
        if not pdb_id:
            continue
        if len(pdb_id.split()) != 1:
            raise ValueError(f"{path}:{line_number}: expected one PDB ID per line.")
        if pdb_id in ids:
            raise ValueError(f"{path}:{line_number}: duplicate PDB ID {pdb_id}.")
        ids.add(pdb_id)
    return ids


def require_no_ids(ids, message):
    if ids:
        examples = ", ".join(sorted(ids)[:10])
        raise ValueError(f"{message}: {len(ids)} IDs ({examples}).")


def filter_indices(args):
    splits = {}
    all_ids = set()
    for split in ("train", "val", "test"):
        ids = read_ids(Path(args.splitDir) / f"{split}_MD.txt")
        if not ids:
            raise ValueError(f"The {split} split is empty.")
        require_no_ids(ids & all_ids, f"The {split} split overlaps another split")
        splits[split] = ids
        all_ids.update(ids)

    excluded = read_ids(args.excludeFile)
    passed = read_ids(args.passedFile)
    test_subset = read_ids(args.testIncludeFile)
    require_no_ids(excluded - all_ids, "Excluded IDs absent from the input splits")
    require_no_ids(passed - all_ids, "Passed IDs absent from the input splits")
    require_no_ids(passed & excluded, "IDs present in both the pass and exclusion lists")
    require_no_ids(
        all_ids - excluded - passed,
        "Split IDs neither excluded nor recorded as passing step 4",
    )
    if not test_subset:
        raise ValueError("The test inclusion list is empty.")
    require_no_ids(test_subset - splits["test"], "Requested test IDs outside the test split")
    require_no_ids(test_subset - passed, "Requested test IDs that did not pass preprocessing")

    with Path(args.indices).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or "pdb_id" not in fieldnames:
            raise ValueError("The input indices CSV must contain a pdb_id column.")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("The input indices CSV has duplicate column names.")
        rows = list(reader)
    indexed_ids = set()
    for row_number, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Indices row {row_number} does not match the CSV header.")
        pdb_id = row["pdb_id"].strip().upper()
        if not pdb_id:
            raise ValueError(f"Indices row {row_number} has an empty pdb_id.")
        indexed_ids.add(pdb_id)
    require_no_ids(indexed_ids - all_ids, "Indexed IDs outside the input splits")
    require_no_ids(passed - indexed_ids, "Passed IDs missing from the input indices")

    retained_ids = {
        "train": splits["train"] & passed,
        "val": splits["val"] & passed,
        "test": test_subset,
    }
    retained_rows = {
        split: [row for row in rows if row["pdb_id"].strip().upper() in ids]
        for split, ids in retained_ids.items()
    }
    summary = {
        "input_complexes": len(all_ids),
        "indexed_complexes": len(indexed_ids),
        "input_interface_rows": len(rows),
        "excluded_complexes": len(excluded),
        "passing_complexes": len(passed),
        "unused_passing_test_complexes": len((splits["test"] & passed) - test_subset),
        "output_complexes": sum(len(ids) for ids in retained_ids.values()),
        "output_interface_rows": sum(len(items) for items in retained_rows.values()),
        "splits": {
            split: {
                "input_complexes": len(splits[split]),
                "passing_complexes": len(splits[split] & passed),
                "output_complexes": len(retained_ids[split]),
                "interface_rows": len(retained_rows[split]),
            }
            for split in splits
        },
    }

    # Validate all input coverage before creating output files. Keep row order,
    # every interface for retained IDs, and each CSV field's value unchanged.
    output = Path(args.outDir)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".filter_indices_", dir=output) as tmpdir:
        temporary = Path(tmpdir)
        for split, items in retained_rows.items():
            with (temporary / f"{split}_MD.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                writer.writeheader()
                writer.writerows(items)
        (temporary / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        for filename in ("train_MD.csv", "val_MD.csv", "test_MD.csv", "summary.json"):
            (temporary / filename).replace(output / filename)
    return summary


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--indices", required=True,
        help="Combined interface CSV from step 3.",
    )
    parser.add_argument(
        "-s", "--splitDir", required=True,
        help="Directory containing the original train_MD.txt, val_MD.txt, and test_MD.txt.",
    )
    parser.add_argument(
        "-e", "--excludeFile", default=directory / "excluded_pdb_list.txt",
        help="Cumulative failures through step 4 (default: the bundled exclusion list).",
    )
    parser.add_argument(
        "-p", "--passedFile", required=True,
        help="Step 4 passed_pdb_ids.txt; must cover exactly the non-excluded split IDs.",
    )
    parser.add_argument(
        "--testIncludeFile", default=directory / "misato_test_include_only_list.txt",
        help="Required test IDs (default: the bundled fixed 100-complex subset).",
    )
    parser.add_argument(
        "-o", "--outDir", required=True,
        help="Output directory for train_MD.csv, val_MD.csv, test_MD.csv, and summary.json.",
    )
    args = parser.parse_args()
    try:
        summary = filter_indices(args)
    except (ValueError, OSError, csv.Error) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
