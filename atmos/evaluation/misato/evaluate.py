"""Evaluate four MISATO metrics for ligand and pocket–ligand trajectories."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from atmos.evaluation.misato.inputs import discover_inputs, load_trajectories, read_ids
from atmos.evaluation.misato.metrics import evaluate_trajectories, RMWD_EXCLUDED_TARGETS


TEST_LIST = Path(__file__).resolve().parents[2] / "data/preprocessing/misato/misato_test_include_only_list.txt"
METRICS = [
    ("rmsf_pearson", "Per-target RMSF r", "median", "Pearson r"),
    ("rmwd", "Root mean W2 distance", "median", "angstrom"),
    ("steric_clashes", "Steric clashes", "mean", "pairs per trajectory"),
    ("t_rmsd_error", "t-RMSD Error", "mean", "angstrom"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--work-dir", type=Path, nargs="+", required=True,
                        help="Run directories, or a parallel-halves launch directory. Inputs must contain disjoint targets.")
    parser.add_argument("-o", "--output-dir", type=Path,
                        help="New or empty output directory; defaults to a timestamped directory under the first input.")
    parser.add_argument("--gt-dir", type=Path,
                        help="Reference PDB/XTC directory; defaults to each input's gt_trajectories/. Not used for pickles.")
    root = Path(os.environ.get("ATMOS_ROOT_DIR", str(Path(__file__).resolve().parents[3] / "atmos_oss")))
    parser.add_argument("--bond-reference-dir", type=Path,
                        default=root / "misato/test_MD",
                        help="Bioassembly pickles containing ligand bonds, required for PDB/XTC inputs. Default: misato/test_MD under ATMOS_ROOT_DIR (repository atmos_oss/ if unset).")
    parser.add_argument("--target-list", type=Path, default=TEST_LIST,
                        help="One PDB ID per line, restricted to the fixed MISATO test set; absent predictions are recorded.")
    parser.add_argument("-n", "--num-workers", type=int, default=6)
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be positive")
    return args


def evaluate_target(task, output, bond_reference_dir):
    pred, ref = load_trajectories(task, output, bond_reference_dir)
    rows = evaluate_trajectories(task["target"], pred, ref)
    ligand_atoms = list(list(pred.top.chains)[-1].atoms)
    bond_key_collisions = len(ligand_atoms) - len({(atom.residue.name, atom.name) for atom in ligand_atoms})
    for row in rows:
        row["ligand_bond_key_collisions"] = bond_key_collisions
    return rows


def build_report(per_target):
    records = []
    for scope in ["ligand", "pocket_ligand"]:
        rows = per_target[per_target.scope == scope]
        for key, label, aggregation, unit in METRICS:
            applicable = rows[~rows.rmwd_excluded] if key == "rmwd" else rows
            values = applicable[key].to_numpy()
            # Undefined correlations remain NaN rather than silently changing
            # the target set for aggregation. Coverage is recorded separately.
            value = getattr(np, aggregation)(values) if len(values) else np.nan
            records.append({"scope": scope, "metric": key, "label": label, "value": value,
                            "aggregation": aggregation, "unit": unit, "n": len(values),
                            "n_finite": int(np.isfinite(values).sum())})
    return pd.DataFrame(records)


def main():
    args = parse_args()
    target_ids = read_ids(args.target_list)
    if not target_ids or target_ids - read_ids(TEST_LIST):
        raise ValueError("The target list must be a nonempty subset of the fixed MISATO test set")
    tasks = discover_inputs(args.work_dir, target_ids, args.gt_dir)
    if any("pred_xtc" in task for task in tasks):
        if args.bond_reference_dir is None:
            raise ValueError("Set --bond-reference-dir or ATMOS_ROOT_DIR for PDB/XTC evaluation")
        for task in tasks:
            if "pred_xtc" in task:
                path = args.bond_reference_dir / f"{task['target']}.pkl.gz"
                if not path.is_file():
                    raise FileNotFoundError(path)
    output = (args.output_dir or args.work_dir[0] / f"misato_evaluation_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    present = {task["target"] for task in tasks}
    manifest = {
        "status": "running", "work_dirs": [str(path.resolve()) for path in args.work_dir],
        "output_dir": str(output), "target_list": str(args.target_list.resolve()),
        "bond_reference_dir": str(args.bond_reference_dir.resolve()) if args.bond_reference_dir else None,
        "targets": sorted(present), "missing_targets": sorted(target_ids - present),
        "inputs": tasks, "frames": 100, "pocket_cutoff_angstrom": 10.0,
        "clash_cutoff_angstrom": 1.1, "cutoff_comparison": "<=",
        "rmwd_excluded_targets": sorted(RMWD_EXCLUDED_TARGETS),
        "num_workers": args.num_workers, "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "scipy", "pandas", "mdtraj", "torch", "biotite"]},
    }
    config_path = output / "run_config.json"
    config_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Evaluation output: {output}")
    print(f"Targets present: {len(present)}/{len(target_ids)}")
    if manifest["missing_targets"]:
        print("Absent predictions: " + ", ".join(manifest["missing_targets"]))

    results, errors = [], {}
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(evaluate_target, task, output, args.bond_reference_dir): task["target"]
                   for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="MISATO"):
            target = futures[future]
            try:
                results.extend(future.result())
            except Exception as exc:
                errors[target] = f"{type(exc).__name__}: {exc}"
    (output / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    manifest["status"] = "failed" if errors else "complete"
    config_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if results:
        per_target = pd.DataFrame(results).sort_values(["target", "scope"])
        per_target.to_csv(output / "per_target.csv", index=False, na_rep="NaN")
    if errors:
        raise RuntimeError(f"Evaluation failed for {len(errors)} targets; see {output / 'errors.json'}")

    report = build_report(per_target)
    report.to_csv(output / "summary.csv", index=False, na_rep="NaN")
    summary = {
        "status": "complete", "target_count": len(present), "missing_targets": manifest["missing_targets"],
        "metrics": json.loads(report.to_json(orient="records", double_precision=15)),
        "rmwd_fallback_targets": {scope: per_target[(per_target.scope == scope) & per_target.rmwd_covariance_fallback].target.tolist()
                                  for scope in ["ligand", "pocket_ligand"]},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    table = report.pivot(index="scope", columns="label", values="value").reindex(
        index=["ligand", "pocket_ligand"], columns=[item[1] for item in METRICS])
    text = (f"Targets evaluated: {len(present)}/{len(target_ids)}; frames per trajectory: 100\n\n"
            + table.to_string(float_format=lambda value: f"{value:.6f}")
            + "\n\nRMWD and t-RMSD Error are in angstroms. Clashes are pairs per trajectory.\n"
            + "RMSF r and RMWD: median over targets. Clashes and t-RMSD Error: mean over targets.\n"
            + "Missing targets, RMWD fallbacks, and finite-value counts are recorded in JSON/CSV outputs.\n")
    (output / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
