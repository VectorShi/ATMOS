# SPDX-License-Identifier: LGPL-2.1-or-later
# Adapted from MISATO src/data/processing/h5_to_traj.py.
# Modified for ATMOS (2026-10-24): batch trajectory conversion, compressed
# topology loading, and worker error handling.
'''MISATO, a database for protein-ligand interactions
    Copyright (C) 2023
                        Till Siebenmorgen  (till.siebenmorgen@helmholtz-munich.de)
                        Sabrina Benassou   (s.benassou@fz-juelich.de)
                        Filipe Menezes     (filipe.menezes@helmholtz-munich.de)
                        Erinç Merdivan     (erinc.merdivan@helmholtz-munich.de)

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA'''


import gzip
import h5py
import argparse
import json
import numpy as np
import pytraj as pt
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool


def open_restart_file(struct, rstPath):
    """
    Open the restart file of the structure.
    Args:
    struct (str): PDB code of the structure
    rstPath (str): Path to the directory of the structures (containing a separate structure folder containing production.rst,production.top files)
    """
    topology = Path(rstPath) / struct / "production.top"
    if not topology.exists():
        compressed = topology.with_suffix(".top.gz")
        temporary = topology.with_suffix(".top.tmp")
        try:
            with gzip.open(str(compressed), "rb") as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            temporary.replace(topology)
        finally:
            if temporary.exists():
                temporary.unlink()
    traj = pt.iterload(os.path.join(rstPath,struct,"production.rst"), str(topology))
    return traj

def create_new_traj(traj, h5_coordinates):
    """
    Coordinates from h5 file are appended as frames to stripped traj.

    Args:
    traj (pytraj trajectory): Trajectory from the restart file
    h5_coordinates (np.array): Coordinates

    """
    h5_coordinates = np.expand_dims(h5_coordinates, axis=2)
    for frameNum in range(np.shape(h5_coordinates)[0]):
        frame = pt.Frame()
        for i in range(traj.n_atoms):
            frame.append_xyz(h5_coordinates[frameNum][i])
        traj.append(frame)
    return traj[1:]

def create_topology(topNameNew, file_top, mask):
    """
    A stripped toplogy is written to file.
    Args:
    topNameNew (str): Name of the new topology file
    file_top (str): Original topology file
    mask (str): Mask to strip the topology

    """
    top = pt.load_topology(file_top)
    top = pt.strip(top, "!({})".format(mask))
    top.save(topNameNew)


def process_one_struct(struct, rstPath, datasetMD, outFilename):
    traj = open_restart_file(struct.lower(), rstPath)

    with h5py.File(datasetMD, 'r') as h5:
        h5_coordinates = h5[struct]['trajectory_coordinates'][:]

    # This mask strips all waters and ions.
    mask = "!:WAT,Na+,Cl-"
    stripped = traj[mask]
    if h5_coordinates.ndim != 3 or h5_coordinates.shape[1:] != (stripped.n_atoms, 3):
        raise ValueError(
            "H5 coordinate shape {} does not match the stripped topology with {} atoms."
            .format(h5_coordinates.shape, stripped.n_atoms)
        )
    if len(h5_coordinates) == 0 or not np.isfinite(h5_coordinates).all():
        raise ValueError("H5 coordinates must contain finite, nonempty frames.")
    new_traj = create_new_traj(stripped, h5_coordinates)
    output = Path(outFilename)
    with tempfile.TemporaryDirectory(prefix="." + struct + "_", dir=str(output.parent)) as tmpdir:
        trajectory = Path(tmpdir) / output.name
        topology = trajectory.with_suffix(".top")
        pt.write_traj(str(trajectory), new_traj, overwrite=True)
        create_topology(str(topology), os.path.join(rstPath, struct.lower(), "production.top"), mask)
        topology.replace(output.with_suffix(".top"))
        trajectory.replace(output)
    return {"pdb_id": struct, "frames": len(h5_coordinates), "atoms": stripped.n_atoms}


def main(args):
    os.makedirs(args.outDir, exist_ok=True)

    def txt_to_list(fpath: str):
        with open(fpath, "r", encoding="utf-8") as f:
            result_list = [line.strip() for line in f if line.strip()]

        return result_list

    splits = ["train_MD.txt", "val_MD.txt", "test_MD.txt"]
    structs = []
    for split_name in splits:
        split_file = os.path.join(args.splitDir, split_name)
        structs = structs + txt_to_list(split_file)

    if len(structs) != len(set(structs)):
        raise ValueError("The split files contain duplicate PDB IDs.")
    if args.num_workers < 1:
        raise ValueError("num_workers must be positive.")

    output = Path(args.outDir)
    started = time.time()
    completed = failed = 0

    def write_summary(status):
        summary = {
            "total": len(structs), "attempted": completed,
            "succeeded": completed - failed, "failed": failed,
            "num_workers": args.num_workers, "status": status,
            "elapsed_seconds": round(time.time() - started, 1),
        }
        temporary = output / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(output / "summary.json")
        return summary

    write_summary("running")
    try:
        with (output / "failed_pdb_ids.txt").open("w") as failure_ids, \
                (output / "failures.jsonl").open("w") as failures, \
                (output / "successes.jsonl").open("w") as successes, \
                ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_one_struct, struct, args.rstPath, args.datasetMD,
                                str(output / (struct + ".nc"))): struct
                for struct in structs
            }
            for future in tqdm(as_completed(futures), total=len(futures), mininterval=10):
                try:
                    result = future.result()
                    successes.write(json.dumps(result) + "\n")
                    successes.flush()
                except BrokenProcessPool:
                    raise
                except Exception as exc:
                    failed += 1
                    failure = {
                        "pdb_id": futures[future], "stage": "step0",
                        "error_type": type(exc).__name__, "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    failure_ids.write(failure["pdb_id"] + "\n")
                    failures.write(json.dumps(failure) + "\n")
                    failure_ids.flush()
                    failures.flush()
                completed += 1
                write_summary("running")
    except BaseException:
        write_summary("interrupted")
        raise
    print(json.dumps(write_summary("completed")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all MISATO split IDs from H5 coordinates to NetCDF trajectories and AMBER topologies."
    )
    parser.add_argument("-s", "--splitDir", required=True, help="Directory containing train/val/test_MD.txt.")
    parser.add_argument("-dMD", "--datasetMD", required=True, help="Path to MD.hdf5.")
    parser.add_argument("-o", "--outDir", required=True, help="Output directory for .nc/.top files, progress, and failures.")
    parser.add_argument("-r", "--rstPath", required=True,
                        help="Directory of lowercase PDB IDs with production.rst and production.top or production.top.gz.")
    parser.add_argument("-n", "--num_workers", required=True, type=int, help="Maximum number of conversion workers.")
    main(parser.parse_args())
