"""Prepare crystal references and full-length GT trajectories for mdCATH."""

import argparse
import gzip
import os
import pickle
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DEFAULT_ROOT = Path(os.environ.get(
    "ATMOS_ROOT_DIR", str(Path(__file__).resolve().parents[3] / "atmos_oss")
))

import mdtraj as md
import pandas as pd
import torch
from biotite.structure.io.pdb import PDBFile
from tqdm import tqdm

from atmos.data.preprocessing.mdcath.load_mdcath import _write_filtered_pdb
from atmos.evaluation.mdcath.trajectories import pred_coord_to_pdb
from protenix.data.pipeline.data_pipeline import DataPipeline
from protenix.data.utils import pdb_to_cif
from protenix.metrics.rmsd import align_pred_to_true


DATASET_RE = re.compile(r"^(mdcath_dataset_[^_]+)_(\d+)\.pkl\.gz$")


def _write_atom_array_pdb(atom_array: Any, save_path: Path) -> None:
    pdb_file = PDBFile()
    pdb_file.set_structure(atom_array)
    pdb_file.write(str(save_path))


def _load_pickle(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def _h5_name_key(path: Path) -> str:
    return path.stem.lower()


def collect_tempo_pickles(tempo_pkl_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(tempo_pkl_dir.glob("*.pkl.gz")):
        match = DATASET_RE.match(path.name)
        if match is None:
            continue
        base = match.group(1)
        grouped.setdefault(base, []).append(path)
    return grouped


def build_h5_map(raw_h5_dir: Path) -> dict[str, Path]:
    return {_h5_name_key(path): path for path in raw_h5_dir.glob("*.h5")}


def process_ref_pdb_task(task: tuple[str, str, str, str, str]) -> tuple[str, str]:
    base, h5_path_str, ref_root_str, tmp_root_str, ref_pdb_dataset = task
    h5_path = Path(h5_path_str)
    ref_root = Path(ref_root_str)
    tmp_root = Path(tmp_root_str)

    out_dir = ref_root / base.removeprefix("mdcath_dataset_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ref.pdb"

    access_code = f"{base}_ref"
    with tempfile.TemporaryDirectory(prefix=f"{base}_ref_", dir=tmp_root) as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_pdb_path = tmp_path / f"{access_code}.pdb"
        filtered_pdb_path = tmp_path / f"{access_code}.filtered.pdb"
        cif_path = tmp_path / f"{access_code}.cif"

        import h5py

        with h5py.File(h5_path, "r") as h5_file:
            code = next(iter(h5_file.keys()))
            if ref_pdb_dataset not in h5_file[code]:
                raise KeyError(f"Missing /{code}/{ref_pdb_dataset} in {h5_path}")
            raw_pdb_path.write_bytes(h5_file[code][ref_pdb_dataset][()])

        _write_filtered_pdb(
            input_pdb_path=raw_pdb_path,
            output_pdb_path=filtered_pdb_path,
        )
        pdb_to_cif(
            input_fname=str(filtered_pdb_path),
            output_fname=str(cif_path),
            entry_id=access_code,
        )
        _, bioassembly_dict = DataPipeline.get_data_from_mmcif(
            mmcif=str(cif_path),
            dataset="Distillation",
        )
        _write_atom_array_pdb(bioassembly_dict["atom_array"], out_path)

    return base, "written"


def _write_aligned_atom_array_pdbs(
    atom_array_stack: list[Any],
    tmp_dir: Path,
    pid: str,
) -> list[str]:
    save_files: list[str] = []
    last_coordinate: torch.Tensor | None = None

    for frame_no, atom_array in enumerate(atom_array_stack):
        coordinate = torch.as_tensor(atom_array.coord, dtype=torch.float32)
        if last_coordinate is not None:
            coordinate = align_pred_to_true(coordinate, last_coordinate)[0]
        last_coordinate = coordinate

        save_file = tmp_dir / f"{pid}_{frame_no:03d}.pdb"
        pred_coord_to_pdb(atom_array, coordinate, str(save_file))
        save_files.append(str(save_file))

    return save_files


def process_gt_trajectory_task(
    task: tuple[str, str, int | None, str]
) -> tuple[str, str, int, int]:
    pkl_path_str, gt_root_str, max_len, tmp_root_str = task
    pkl_path = Path(pkl_path_str)
    gt_root = Path(gt_root_str)
    tmp_root = Path(tmp_root_str)

    match = DATASET_RE.match(pkl_path.name)
    if match is None:
        raise ValueError(f"Unexpected mdCATH tempo filename: {pkl_path}")
    pid = f"{match.group(1)}_{match.group(2)}"

    xtc_path = gt_root / f"{pid}.xtc"
    pdb_path = gt_root / f"{pid}.pdb"

    batch = _load_pickle(pkl_path)
    atom_array_stack = batch["atom_array_stack"]
    if max_len is not None:
        atom_array_stack = atom_array_stack[:max_len]
    if len(atom_array_stack) == 0:
        raise ValueError(f"No frames found in {pkl_path}")

    with tempfile.TemporaryDirectory(prefix=f"{pid}_gt_", dir=tmp_root) as tmp_dir:
        save_files = _write_aligned_atom_array_pdbs(atom_array_stack, Path(tmp_dir), pid)
        traj = md.load(save_files)

    traj.save(str(xtc_path))
    traj[0].save_pdb(str(pdb_path))
    return pid, "written", traj.n_frames, traj.n_atoms


def _residue_signature(topology: md.Topology) -> list[tuple[int, int, str]]:
    return [
        (residue.chain.index, residue.resSeq, residue.name)
        for residue in topology.residues
    ]


def _sequence(topology: md.Topology) -> list[str]:
    return [residue.name for residue in topology.residues]


def _atom_reprs(topology: md.Topology) -> list[str]:
    return [repr(atom) for atom in topology.atoms]


def verify_one(base: str, ref_root: Path, gt_root: Path, pred_dir: Path | None) -> dict[str, Any]:
    dname = base.removeprefix("mdcath_dataset_")
    ref_pdb = ref_root / dname / "ref.pdb"
    gt_pdb = gt_root / f"{base}_0.pdb"
    pred_pdb = pred_dir / f"{base}_0.pdb" if pred_dir is not None else None

    row: dict[str, Any] = {
        "base": base,
        "dname": dname,
        "ref_pdb": str(ref_pdb),
        "gt_pdb": str(gt_pdb),
        "pred_pdb": str(pred_pdb) if pred_pdb is not None else "",
        "ref_exists": ref_pdb.exists(),
        "gt_exists": gt_pdb.exists(),
        "pred_exists": pred_pdb.exists() if pred_pdb is not None else False,
    }
    if not ref_pdb.exists() or not gt_pdb.exists():
        return row

    ref_top = md.load(str(ref_pdb)).topology
    gt_top = md.load(str(gt_pdb)).topology
    row.update(
        {
            "ref_atoms": ref_top.n_atoms,
            "gt_atoms": gt_top.n_atoms,
            "ref_residues": ref_top.n_residues,
            "gt_residues": gt_top.n_residues,
            "ref_gt_sequence_match": _sequence(ref_top) == _sequence(gt_top),
            "ref_gt_chain_resid_match": _residue_signature(ref_top)
            == _residue_signature(gt_top),
            "ref_gt_atom_repr_intersection": len(
                set(_atom_reprs(ref_top)).intersection(_atom_reprs(gt_top))
            ),
        }
    )

    if pred_pdb is not None and pred_pdb.exists():
        pred_top = md.load(str(pred_pdb)).topology
        row.update(
            {
                "pred_atoms": pred_top.n_atoms,
                "pred_residues": pred_top.n_residues,
                "gt_pred_sequence_match": _sequence(gt_top) == _sequence(pred_top),
                "gt_pred_chain_resid_match": _residue_signature(gt_top)
                == _residue_signature(pred_top),
                "gt_pred_atom_repr_intersection": len(
                    set(_atom_reprs(gt_top)).intersection(_atom_reprs(pred_top))
                ),
                "ref_pred_atom_repr_intersection": len(
                    set(_atom_reprs(ref_top)).intersection(_atom_reprs(pred_top))
                ),
            }
        )

    return row


def run_ref_pdb_generation(
    grouped_pickles: dict[str, list[Path]],
    h5_map: dict[str, Path],
    ref_root: Path,
    tmp_root: Path,
    max_workers: int,
    ref_pdb_dataset: str,
) -> None:
    tasks = []
    missing = []
    for base in sorted(grouped_pickles):
        h5_path = h5_map.get(base.lower())
        if h5_path is None:
            missing.append(base)
            continue
        tasks.append((base, str(h5_path), str(ref_root), str(tmp_root), ref_pdb_dataset))
    if missing:
        raise FileNotFoundError(f"Missing raw H5 files for {len(missing)} bases: {missing[:8]}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_ref_pdb_task, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="ref pdb"):
            future.result()


def run_gt_generation(
    grouped_pickles: dict[str, list[Path]],
    gt_root: Path,
    tmp_root: Path,
    max_len: int | None,
    max_workers: int,
) -> None:
    tasks = [
        (str(path), str(gt_root), max_len, str(tmp_root))
        for paths in grouped_pickles.values()
        for path in sorted(paths)
    ]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_gt_trajectory_task, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="gt trajectories"):
            future.result()


def write_verification(
    grouped_pickles: dict[str, list[Path]],
    ref_root: Path,
    gt_root: Path,
    pred_dir: Path | None,
    output_path: Path,
) -> pd.DataFrame:
    rows = [
        verify_one(base, ref_root=ref_root, gt_root=gt_root, pred_dir=pred_dir)
        for base in sorted(grouped_pickles)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-h5-dir",
        type=Path,
        required=True,
        help="Directory containing the raw mdCATH H5 files.",
    )
    parser.add_argument(
        "--tempo-pkl-dir",
        type=Path,
        default=DEFAULT_ROOT / "mdCATH_tempo/mdCATH_tempo_test",
        help="Processed test pickles; defaults to mdCATH_tempo/mdCATH_tempo_test under ATMOS_ROOT_DIR (repository atmos_oss/ if unset).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ROOT / "mdCATH_tempo",
        help="Reference output root; defaults to mdCATH_tempo under ATMOS_ROOT_DIR (repository atmos_oss/ if unset).",
    )
    parser.add_argument(
        "--pred-trajectories-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Optional GT trajectory frame cap. Omit to keep all frames.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--ref-pdb-dataset",
        choices=["pdbProteinAtoms", "pdb"],
        default="pdbProteinAtoms",
        help="H5 dataset used for ref.pdb; coordinates are taken directly from this PDB.",
    )
    parser.add_argument("--skip-ref-pdb", action="store_true")
    parser.add_argument("--skip-gt-trajectories", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ref_root = args.output_root / "mdCATH_tempo_test_ref_pdb"
    gt_root = args.output_root / "mdCATH_tempo_test_gt_trajectories"
    tmp_root = args.output_root / "tmp_prepare_mdcath_ensemble_inputs"
    ref_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    grouped_pickles = collect_tempo_pickles(args.tempo_pkl_dir)
    if not grouped_pickles:
        raise RuntimeError(f"No mdCATH tempo pickles found in {args.tempo_pkl_dir}")
    h5_map = build_h5_map(args.raw_h5_dir)
    print(f"Found {len(grouped_pickles)} mdCATH bases and {sum(len(v) for v in grouped_pickles.values())} replicas.")

    if not args.skip_ref_pdb:
        run_ref_pdb_generation(
            grouped_pickles=grouped_pickles,
            h5_map=h5_map,
            ref_root=ref_root,
            tmp_root=tmp_root,
            max_workers=args.max_workers,
            ref_pdb_dataset=args.ref_pdb_dataset,
        )

    if not args.skip_gt_trajectories:
        run_gt_generation(
            grouped_pickles=grouped_pickles,
            gt_root=gt_root,
            tmp_root=tmp_root,
            max_len=args.max_len,
            max_workers=args.max_workers,
        )

    verification_path = args.output_root / "mdCATH_tempo_test_ensemble_input_check.csv"
    df = write_verification(
        grouped_pickles=grouped_pickles,
        ref_root=ref_root,
        gt_root=gt_root,
        pred_dir=args.pred_trajectories_dir,
        output_path=verification_path,
    )
    print(f"Wrote {verification_path}")
    if {"ref_gt_sequence_match", "ref_gt_chain_resid_match"}.issubset(df.columns):
        print(
            "ref vs gt:",
            f"sequence={int(df['ref_gt_sequence_match'].sum())}/{len(df)}",
            f"chain_resid={int(df['ref_gt_chain_resid_match'].sum())}/{len(df)}",
        )
    if "gt_pred_sequence_match" in df.columns:
        pred_rows = df[df["pred_exists"] == True]
        print(
            "gt vs pred:",
            f"sequence={int(pred_rows['gt_pred_sequence_match'].sum())}/{len(pred_rows)}",
            f"chain_resid={int(pred_rows['gt_pred_chain_resid_match'].sum())}/{len(pred_rows)}",
        )


if __name__ == "__main__":
    main()
