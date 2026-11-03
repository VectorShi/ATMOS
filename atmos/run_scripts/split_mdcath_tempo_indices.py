#!/usr/bin/env python
"""Split mdCATH tempo inference indices into two CSV files."""

import argparse
import csv
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    atmos_root = Path(os.environ.get("ATMOS_ROOT_DIR", str(Path(__file__).resolve().parents[2] / "atmos_oss")))
    default_input = atmos_root / "mdCATH_tempo/mdCATH_tempo_test.csv"
    default_output_dir = Path(__file__).resolve().parents[2] / "output/mdcath_inference/indices"

    parser = argparse.ArgumentParser(
        description="Split the mdCATH tempo test indices into two balanced halves."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=default_input,
        help="Source mdCATH tempo indices CSV; defaults to the test index under ATMOS_ROOT_DIR (repository atmos_oss/ if unset).",
    )
    parser.add_argument(
        "--part0-csv",
        type=Path,
        default=default_output_dir / "mdCATH_tempo_test_part0.csv",
        help="Output CSV for the first half.",
    )
    parser.add_argument(
        "--part1-csv",
        type=Path,
        default=default_output_dir / "mdCATH_tempo_test_part1.csv",
        help="Output CSV for the second half.",
    )
    parser.add_argument(
        "--pdb-id-column",
        default="pdb_id",
        help="Column used for mdCATH inference filtering.",
    )
    parser.add_argument(
        "--pdb-id-suffix",
        default="_0",
        help="Keep only rows whose pdb_id ends with this suffix. Use an empty value to disable.",
    )
    return parser.parse_args()


def read_indices(
    input_csv: Path,
    pdb_id_column: str,
) -> tuple[list[str], list[dict[str, str]]]:
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {input_csv}")
        if pdb_id_column not in reader.fieldnames:
            raise ValueError(
                f"Input CSV {input_csv} has no column {pdb_id_column!r}. "
                f"Available columns: {reader.fieldnames}"
            )
        return reader.fieldnames, list(reader)


def write_indices(
    output_csv: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_indices(args.input_csv, args.pdb_id_column)

    filtered_rows = rows
    if args.pdb_id_suffix:
        filtered_rows = [
            row
            for row in rows
            if row[args.pdb_id_column].endswith(args.pdb_id_suffix)
        ]

    if not filtered_rows:
        raise ValueError(
            f"No rows left after filtering {args.pdb_id_column} "
            f"by suffix {args.pdb_id_suffix!r}"
        )

    # Keep all chain/interface rows for a protein in the same inference group.
    pdb_ids = list(dict.fromkeys(row[args.pdb_id_column] for row in filtered_rows))
    if len(pdb_ids) < 2:
        raise ValueError("At least two distinct pdb_ids are required for two halves")
    split_at = (len(pdb_ids) + 1) // 2
    part0_ids = set(pdb_ids[:split_at])
    part0_rows = [row for row in filtered_rows if row[args.pdb_id_column] in part0_ids]
    part1_rows = [row for row in filtered_rows if row[args.pdb_id_column] not in part0_ids]

    write_indices(args.part0_csv, fieldnames, part0_rows)
    write_indices(args.part1_csv, fieldnames, part1_rows)

    print(
        "split mdCATH tempo indices: "
        f"source_rows={len(rows)} filtered_rows={len(filtered_rows)} "
        f"proteins={len(pdb_ids)} part0_proteins={split_at} part1_proteins={len(pdb_ids) - split_at} "
        f"part0={len(part0_rows)} part1={len(part1_rows)} "
        f"part0_csv={args.part0_csv} part1_csv={args.part1_csv}"
    )


if __name__ == "__main__":
    main()
