"""Evaluate mdCATH trajectories and save aggregate and per-protein metrics."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys

import mdtraj as md
import numpy as np
import pandas as pd
from tqdm import tqdm

from atmos.evaluation.mdcath.dynamics import evaluate_protein
from atmos.evaluation.mdcath.report import build_report


REPO = Path(__file__).resolve().parents[3]
EVALUATION = Path(__file__).resolve().parent
# Exclude these targets because topology inconsistencies bias the evaluation:
# 1waae00: crystal and GT/prediction chain/residue numbering differ; the current
# ensemble alignment retains only 6 CAs, including 3 incorrect correspondences.
# 2noga01: the prediction topology omits UNK43; treating residues 42 and 44 as
# adjacent introduces false bond breaks and dihedrals, while temporal RMSD uses
# different CA selections for GT and prediction.
EXCLUDED_PROTEINS = {"1waae00", "2noga01"}
DEFAULT_DATA = Path(os.environ.get("ATMOS_ROOT_DIR", str(REPO / "atmos_oss"))) / "mdCATH_tempo"


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--work-dir", type=Path, required=True,
                        help="Inference directory containing predictions/ or trajectories/.")
    parser.add_argument("-o", "--output-dir", type=Path,
                        help="New or empty output directory; default: a timestamped subdirectory of WORK_DIR.")
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_DATA / "mdCATH_tempo_test_gt_trajectories",
                        help="GT trajectories; default: mdCATH_tempo/mdCATH_tempo_test_gt_trajectories under ATMOS_ROOT_DIR (repository atmos_oss/ if unset).")
    parser.add_argument("--ref-dir", type=Path, default=DEFAULT_DATA / "mdCATH_tempo_test_ref_pdb",
                        help="Crystal references; default: mdCATH_tempo/mdCATH_tempo_test_ref_pdb under ATMOS_ROOT_DIR (repository atmos_oss/ if unset).")
    parser.add_argument("--max-len", type=positive_int, default=400,
                        help="Generated frames to evaluate. Existing XTC files must have exactly this length.")
    parser.add_argument("--ensemble-workers", type=positive_int, default=6)
    parser.add_argument("--dynamics-workers", type=positive_int, default=16)
    parser.add_argument("--skip-trajectory-conversion", action="store_true",
                        help="Use existing trajectories even when predictions/ is present.")
    return parser.parse_args()


def run_logged(script, arguments, log_path):
    command = [sys.executable, str(script), *map(str, arguments)]
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{script.name} exited with {result.returncode}; see {log_path}")


def stage_inputs(args, output):
    use_predictions = not args.skip_trajectory_conversion and (args.work_dir / "predictions").is_dir()
    source = args.work_dir / ("predictions" if use_predictions else "trajectories")
    if not source.is_dir():
        raise FileNotFoundError(source)
    suffix = "*.pkl.gz" if use_predictions else "*.xtc"
    selected = {}
    excluded = set()
    for path in sorted(source.glob(suffix)):
        match = re.search(r"mdcath_dataset_(.+)_0(?:\.pkl\.gz|\.xtc)$", path.name)
        if not match:
            continue
        protein = match[1]
        if protein in EXCLUDED_PROTEINS:
            excluded.add(protein)
            continue
        if protein in selected:
            raise ValueError(f"Duplicate prediction for {protein}")
        selected[protein] = path.resolve()
    if len(selected) < 2:
        raise ValueError("At least two proteins are needed for the correlation metrics")

    # Check all required reference files before starting the expensive stages.
    for protein, path in selected.items():
        stem = f"mdcath_dataset_{protein}_0"
        required = [args.gt_dir / f"mdcath_dataset_{protein}_{replica}.xtc" for replica in range(5)]
        required += [args.gt_dir / f"{stem}.pdb", args.ref_dir / protein / "ref.pdb"]
        if not use_predictions:
            required.append(path.with_suffix(".pdb"))
        for required_path in required:
            if not required_path.is_file():
                raise FileNotFoundError(required_path)

    target = output / source.name
    target.mkdir()
    for protein, path in selected.items():
        (target / path.name).symlink_to(path)
        if not use_predictions:
            (target / path.with_suffix(".pdb").name).symlink_to(path.with_suffix(".pdb"))
    if use_predictions:
        print("Converting predictions to trajectories...", flush=True)
        run_logged(EVALUATION / "trajectories.py",
                   ["--work_dir", output, "--max_len", args.max_len], output / "conversion.log")

    trajectories = output / "trajectories"
    actual = {path.name[len("mdcath_dataset_"):-len("_0.xtc")] for path in trajectories.glob("*_0.xtc")}
    if actual != set(selected):
        raise RuntimeError(f"Converted trajectory coverage mismatch: missing={set(selected) - actual}, extra={actual - set(selected)}")
    for protein in selected:
        stem = f"mdcath_dataset_{protein}_0"
        if not (trajectories / f"{stem}.pdb").is_file():
            raise FileNotFoundError(trajectories / f"{stem}.pdb")
        with md.open(str(trajectories / f"{stem}.xtc")) as reader:
            frames = len(reader)
        if frames != args.max_len:
            raise ValueError(f"{protein}: found {frames} frames, expected {args.max_len}")
    return sorted(selected), sorted(excluded), use_predictions


def main():
    args = parse_args()
    args.work_dir, args.gt_dir, args.ref_dir = (path.resolve() for path in (args.work_dir, args.gt_dir, args.ref_dir))
    output = args.output_dir or args.work_dir / f"mdcath_evaluation_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation output: {output}", flush=True)
    proteins, excluded, converted = stage_inputs(args, output)
    manifest = {
        "work_dir": str(args.work_dir), "output_dir": str(output),
        "gt_dir": str(args.gt_dir), "ref_dir": str(args.ref_dir),
        "max_len": args.max_len, "proteins": proteins,
        "excluded_proteins": sorted(EXCLUDED_PROTEINS), "excluded_present_in_input": excluded,
        "converted_predictions": converted,
        "ensemble_workers": args.ensemble_workers, "dynamics_workers": args.dynamics_workers,
        "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in
            ["numpy", "pandas", "scipy", "mdtraj", "deeptime", "statsmodels", "torch"]},
    }
    (output / "run_config.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Computing CA-only ensemble metrics for {len(proteins)} proteins...", flush=True)
    run_logged(EVALUATION / "ensemble.py", [
        "--result_dir", output / "trajectories", "--gt_dir", args.gt_dir,
        "--ref_dir", args.ref_dir, "--ca_only", "--num_workers", args.ensemble_workers,
    ], output / "ensemble.log")
    with (output / "out_ca_only.pkl").open("rb") as handle:
        ensemble_data = pickle.load(handle)
    if set(ensemble_data) != set(proteins):
        raise RuntimeError(f"Missing ensemble results: {set(proteins) - set(ensemble_data)}; see ensemble.log")

    print("Computing RMSD time-series, val_ca, MSM transition RMSE, and TIC0 decorrelation...", flush=True)
    results, failures = [], {}
    with ProcessPoolExecutor(max_workers=args.dynamics_workers) as executor:
        futures = {executor.submit(evaluate_protein, protein, output / "trajectories", args.gt_dir, args.max_len): protein
                   for protein in proteins}
        for future in tqdm(as_completed(futures), total=len(futures)):
            protein = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures[protein] = f"{type(exc).__name__}: {exc}"
    (output / "errors.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"Dynamics evaluation failed for {len(failures)} proteins; see {output / 'errors.json'}")

    dynamics = pd.DataFrame(results).set_index("protein").sort_index()
    summary, per_protein = build_report(ensemble_data, dynamics)
    per_protein.to_csv(output / "per_protein.csv", na_rep="NaN")
    summary.to_csv(output / "summary.csv", index=False, na_rep="NaN")
    metric_records = summary.replace([np.inf, -np.inf], np.nan).to_json(orient="records", double_precision=15)
    notes = [
        "n counts contributing proteins; pooled-residue metrics also report the number of input proteins.",
        "std is the sample standard deviation (ddof=1) for val_ca and msm_tm_rmse only.",
        "Exposed residue J and Exposed MI matrix rho are undefined in CA-only sidechain analysis; NaN is retained.",
        "TIC0 reference decay uses the first max_len frames of concatenated GT, threshold <0.5, and at most 100 lags.",
        "When predicted TIC0 does not cross the threshold, its decay time is max_len.",
    ]
    (output / "summary.json").write_text(json.dumps({"status": "complete", "protein_count": len(proteins),
        "metrics": json.loads(metric_records), "notes": notes}, indent=2, allow_nan=False) + "\n")
    table = summary.to_string(index=False, float_format=lambda value: f"{value:.6f}")
    header = f"Proteins evaluated: {len(proteins)}"
    (output / "summary.txt").write_text(header + "\n\n" + table + "\n\n" + "\n".join(notes) + "\n")
    print(header)
    print(table)
    print(f"Saved summary.csv, summary.json, summary.txt, and per_protein.csv in {output}")


if __name__ == "__main__":
    main()
