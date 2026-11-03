"""Preprocess mdCATH trajectories into trajectory-level bioassembly pickles.

High-level workflow:
1. Read one mdCATH H5 file and extract the shared topology plus replica coordinates.
2. For each replica, iterate over all frames sequentially.
3. For each frame, save a temporary PDB, remove hydrogens and waters, convert the
   filtered structure to mmCIF, and run ``DataPipeline.get_data_from_mmcif()``
   with the ``Distillation`` dataset path.
4. Keep the first frame's ``bioassembly_dict`` as the output template and append
   every frame's ``atom_array`` into ``atom_array_stack``.
"""
import argparse
import csv
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[4]))

import h5py
import mdtraj as md
import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile
from tqdm import tqdm

from protenix.data.core.filter import Filter
from protenix.data.pipeline.data_pipeline import DataPipeline
from protenix.data.utils import pdb_to_cif
from protenix.utils.file_io import dump_gzip_pickle


def _extract_structure_and_coordinates(
    h5_file: h5py.File,
    code: str,
    temperature: str,
    replica_id: int,
) -> tuple[str, np.ndarray]:
    """Extract the raw topology PDB and coordinates for one replica.

    Args:
        h5_file: Opened mdCATH H5 handle.
        code: Top-level group name inside the H5 file.
        temperature: Temperature key used by the dataset.
        replica_id: Replica index under the temperature group.

    Returns:
        A tuple containing:
        - The temporary PDB file path holding the shared topology.
        - The coordinates array in angstrom with shape ``(T, N_atom, 3)``.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as pdb_file:
        pdb_file.write(h5_file[code]["pdbProteinAtoms"][()])
        pdb_file.flush()
        coords = h5_file[code][str(temperature)][str(replica_id)]["coords"][:]

    return pdb_file.name, coords


def convert_to_mdtraj(h5_file: h5py.File, temperature: str, replica_id: int) -> md.Trajectory:
    """Convert one replica in the mdCATH H5 file into an ``mdtraj.Trajectory``.

    Args:
        h5_file: Opened mdCATH H5 handle.
        temperature: Temperature key used by the dataset.
        replica_id: Replica index under the temperature group.

    Returns:
        A trajectory with the shared topology and per-frame coordinates.
    """
    code = next(iter(h5_file.keys()))
    pdb_path, coords = _extract_structure_and_coordinates(
        h5_file=h5_file,
        code=code,
        temperature=temperature,
        replica_id=replica_id,
    )

    try:
        trajectory = md.load(pdb_path)
    finally:
        os.unlink(pdb_path)

    trajectory.xyz = coords.copy() / 10.0 # mdtraj stores coordinates in nm units.
    trajectory.time = np.arange(1, coords.shape[0] + 1)
    return trajectory


def _write_filtered_pdb(input_pdb_path: Path, output_pdb_path: Path) -> None:
    """Load a frame PDB, remove hydrogens and waters, and save the filtered PDB.

    Args:
        input_pdb_path: Temporary PDB path written by ``mdtraj`` for one frame.
        output_pdb_path: Output path for the filtered PDB.
    """
    pdb_file = PDBFile.read(str(input_pdb_path))
    atom_array = pdb_file.get_structure(model=1, include_bonds=True, altloc="first")

    atom_array = Filter.remove_hydrogens(atom_array)
    atom_array = Filter.remove_water(atom_array)

    filtered_pdb = PDBFile()
    filtered_pdb.set_structure(atom_array)
    with open(output_pdb_path, "w", encoding="utf-8") as handle:
        filtered_pdb.write(handle)


def frame_to_bioassembly(
    trajectory: md.Trajectory,
    frame_id: int,
    access_code: str,
    tmp_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Process a single trajectory frame into a ``bioassembly_dict``.

    The frame is written to a temporary PDB, filtered to remove hydrogens and
    waters, converted to mmCIF through the shared helper, and finally parsed by
    ``DataPipeline.get_data_from_mmcif()``.

    Args:
        trajectory: The full trajectory for one replica.
        frame_id: Frame index to process.
        access_code: Output naming prefix for this replica.
        tmp_dir: Replica-scoped temporary directory.

    Returns:
        ``(sample_indices_list, bioassembly_dict)`` for the frame.
    """
    raw_pdb_path = tmp_dir / f"{access_code}_frame{frame_id}.pdb"
    filtered_pdb_path = tmp_dir / f"{access_code}_frame{frame_id}.filtered.pdb"
    cif_path = tmp_dir / f"{access_code}_frame{frame_id}.cif"

    trajectory[frame_id].save_pdb(str(raw_pdb_path))

    try:
        _write_filtered_pdb(
            input_pdb_path=raw_pdb_path,
            output_pdb_path=filtered_pdb_path,
        )
        pdb_to_cif(
            input_fname=str(filtered_pdb_path),
            output_fname=str(cif_path),
            entry_id=access_code,
        )
        sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
            mmcif=str(cif_path),
            dataset="Distillation",
        )
    finally:
        for tmp_path in (raw_pdb_path, filtered_pdb_path, cif_path):
            if tmp_path.exists():
                tmp_path.unlink()

    return sample_indices_list, bioassembly_dict


def process_replica(
    h5_fpath: str,
    temperature: str,
    replica_id: int,
    save_dir: str,
    tmp_root: str,
) -> list[dict[str, Any]]:
    """Process one replica and persist a trajectory-level bioassembly pickle.

    The saved object is the first frame's ``bioassembly_dict`` augmented with an
    ``atom_array_stack`` field containing every frame's atom array.

    Args:
        h5_fpath: Path to the mdCATH H5 file.
        temperature: Temperature key used by the dataset.
        replica_id: Replica index to process.
        save_dir: Directory where the output pickle will be saved.
        tmp_root: Root directory used for replica-scoped temporary files.

    Returns:
        The first frame's ``sample_indices_list`` for the dataset CSV.
    """
    access_code = f"{Path(h5_fpath).stem.lower()}_{replica_id}"

    with h5py.File(h5_fpath, "r") as h5_file:
        trajectory = convert_to_mdtraj(
            h5_file=h5_file,
            temperature=temperature,
            replica_id=replica_id,
        )

    first_frame_indices: list[dict[str, Any]] = []
    template_bioassembly_dict: dict[str, Any] | None = None
    atom_array_stack: list[AtomArray] = []

    with tempfile.TemporaryDirectory(
        prefix=f"{access_code}_",
        dir=tmp_root,
    ) as replica_tmp_dir:
        replica_tmp_path = Path(replica_tmp_dir)

        for frame_id in range(trajectory.n_frames):
            sample_indices_list, bioassembly_dict = frame_to_bioassembly(
                trajectory=trajectory,
                frame_id=frame_id,
                access_code=access_code,
                tmp_dir=replica_tmp_path,
            )

            if frame_id == 0:
                first_frame_indices = sample_indices_list
                template_bioassembly_dict = bioassembly_dict

            atom_array_stack.append(bioassembly_dict["atom_array"])

    if template_bioassembly_dict is None:
        raise RuntimeError(f"No frames were processed for replica {access_code}")

    # Store the first frame's metadata together with all trajectory frames.
    template_bioassembly_dict["atom_array_stack"] = atom_array_stack

    save_path = Path(save_dir) / f"{access_code}.pkl.gz"
    dump_gzip_pickle(template_bioassembly_dict, save_path)
    return first_frame_indices


def process_traj(
    h5_fpath: str,
    temperature: str,
    save_dir: str,
    tmp_root: str,
    num_replicas: int = 5,
) -> list[dict[str, Any]]:
    """Process all replicas for one mdCATH H5 file.

    Args:
        h5_fpath: Path to the mdCATH H5 file.
        temperature: Temperature key used by the dataset.
        save_dir: Directory where trajectory pickles will be written.
        tmp_root: Root directory used for temporary frame files.
        num_replicas: Number of replicas to process for each trajectory file.

    Returns:
        A merged list of sample indices from the first frame of each replica.
    """
    all_indices_list: list[dict[str, Any]] = []
    for replica_id in range(num_replicas):
        first_frame_indices = process_replica(
            h5_fpath=h5_fpath,
            temperature=temperature,
            replica_id=replica_id,
            save_dir=save_dir,
            tmp_root=tmp_root,
        )
        if first_frame_indices:
            all_indices_list.extend(first_frame_indices)
    return all_indices_list


def process_mdcath_multiprocess(
    h5_files: list[str],
    bioassembly_dir: str,
    indices_file: str,
    temperature: str,
    tmp_root: str,
    max_workers: int | None = None,
    num_replicas: int = 5,
) -> None:
    """Process multiple mdCATH H5 files in parallel and write the CSV index.

    Args:
        h5_files: List of mdCATH H5 paths.
        bioassembly_dir: Directory where trajectory pickles will be written.
        indices_file: CSV path for the merged sample indices.
        temperature: Temperature key used by the dataset.
        tmp_root: Root directory used for temporary frame files.
        max_workers: Number of worker processes for file-level parallelism.
        num_replicas: Number of replicas to process for each H5 file.
    """
    all_indices_list: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_traj,
                h5_fpath,
                temperature,
                bioassembly_dir,
                tmp_root,
                num_replicas,
            ): h5_fpath
            for h5_fpath in h5_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                sample_indices_list = future.result()
                if sample_indices_list:
                    all_indices_list.extend(sample_indices_list)
            except Exception as exc:
                failed_h5 = futures[future]
                print(f"[ERROR] Failed processing {failed_h5}: {exc}")

    pd.DataFrame(all_indices_list).to_csv(
        indices_file,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for mdCATH preprocessing."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name to assign to the preprocessed dataset.",
    )
    parser.add_argument(
        "--fpath_list",
        type=str,
        required=True,
        help=(
            "Path to a text file containing raw mdCATH H5 paths, one path per line."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=str,
        required=True,
        help="Temperature (in Kelvin) used in the MD simulation.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory where preprocessed data will be saved.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Number of worker processes used for file-level parallelism.",
    )
    parser.add_argument(
        "--num_replicas",
        type=int,
        default=5,
        help="Number of replicas to process for each mdCATH H5 file.",
    )
    parser.add_argument(
        "--tmp_root",
        type=str,
        default="af3-dev/mdCATH/tmp",
        help="Root directory for temporary frame files.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for mdCATH preprocessing."""
    args = get_args()

    tmp_root = Path(args.tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    with open(args.fpath_list, "r", encoding="utf-8") as handle:
        raw_paths = [line.strip() for line in handle if line.strip()]

    valid_paths = [path for path in raw_paths if Path(path).exists()]
    print(f"Found {len(valid_paths)} valid data paths out of {len(raw_paths)} entries.")

    indices_save_path = Path(args.save_dir) / f"{args.dataset_name}.csv"
    bioassembly_save_dir = Path(args.save_dir) / args.dataset_name
    bioassembly_save_dir.mkdir(parents=True, exist_ok=True)

    if indices_save_path.exists():
        print("Already preprocessed. Exit.")
        return

    process_mdcath_multiprocess(
        h5_files=valid_paths,
        bioassembly_dir=str(bioassembly_save_dir),
        indices_file=str(indices_save_path),
        temperature=args.temperature,
        tmp_root=str(tmp_root),
        max_workers=args.max_workers,
        num_replicas=args.num_replicas,
    )


if __name__ == "__main__":
    main()
