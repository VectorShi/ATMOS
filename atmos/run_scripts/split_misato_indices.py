#!/usr/bin/env python
"""Split MISATO test complexes into two groups, keeping their interface rows together."""

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--part0-csv", type=Path, required=True)
    parser.add_argument("--part1-csv", type=Path, required=True)
    args = parser.parse_args()

    include_file = (
        Path(__file__).resolve().parents[1]
        / "data/preprocessing/misato/misato_test_include_only_list.txt"
    )
    included_ids = {
        line.strip().lower()
        for line in include_file.read_text().splitlines()
        if line.strip()
    }
    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or "pdb_id" not in fieldnames:
            raise ValueError("The input CSV must contain a pdb_id column")
        rows = list(reader)

    # Apply the inference inclusion list before splitting by complex count.
    filtered_rows = [row for row in rows if row["pdb_id"] in included_ids]
    pdb_ids = list(dict.fromkeys(row["pdb_id"] for row in filtered_rows))
    if len(pdb_ids) < 2:
        raise ValueError("At least two included complexes are required for two groups")
    paths = [args.input_csv.resolve(), args.part0_csv.resolve(), args.part1_csv.resolve()]
    if len(set(paths)) != 3:
        raise ValueError("Input and output CSV paths must be distinct")

    split_at = (len(pdb_ids) + 1) // 2
    groups = [set(pdb_ids[:split_at]), set(pdb_ids[split_at:])]
    print(f"MISATO indices: source_rows={len(rows)}, test_rows={len(filtered_rows)}, complexes={len(pdb_ids)}")
    for part, (ids, output) in enumerate(zip(groups, [args.part0_csv, args.part1_csv])):
        part_rows = [row for row in filtered_rows if row["pdb_id"] in ids]
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(part_rows)
        print(f"part{part}: complexes={len(ids)}, rows={len(part_rows)}, csv={output}")


if __name__ == "__main__":
    main()
