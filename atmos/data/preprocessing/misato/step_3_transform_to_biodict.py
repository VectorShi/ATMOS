"""Convert MISATO trajectories to bioassembly dictionaries and ligand-interface indices.

Each pickle contains the first frame's metadata and all frames in
``atom_array_stack``. Interface indices are taken from the first frame and
must include the ligand chain.
Progress, completed IDs with interface indices, and failures are recorded in
the output directory as each complex finishes.
"""

import os
import sys
import argparse
import json
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[4]))

import mdtraj as md
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv
from protenix.data.utils import pdb_to_cif
from protenix.data.pipeline.data_pipeline import DataPipeline
from protenix.utils.file_io import dump_gzip_pickle


def json_scalar(value):
    """Convert NumPy index-field scalars to JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON.")


def frame_to_bioassembly_dict(
    traj,
    frame_no: int,
    access_code: str,
    tmp_dir: str,
):
    pdb_fpath = os.path.join(
        tmp_dir, f"{access_code}.pdb"
    )
    cif_fpath = pdb_fpath[:-4] + ".cif"

    try:
        traj[frame_no].save_pdb(pdb_fpath)
        pdb_to_cif(pdb_fpath, cif_fpath, entry_id=access_code)
        sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
            mmcif=cif_fpath,
            dataset="Distillation",
        )
        if bioassembly_dict.get("atom_array") is None or "token_array" not in bioassembly_dict:
            raise ValueError(
                f"{access_code} frame {frame_no}: mmCIF processing returned no usable atom/token arrays."
            )
    finally:
        for path in (pdb_fpath, cif_fpath):
            if os.path.exists(path):
                os.unlink(path)

    return sample_indices_list, bioassembly_dict


def process_one_struct(struct, trajDir, outDir, tmpdir):
    # load traj
    topfile = os.path.join(trajDir, f"{struct}.pdb")
    xtcfile = os.path.join(trajDir, f"{struct}.xtc")

    traj = md.load(xtcfile, top = topfile)

    access_code = struct.lower()
    with tempfile.TemporaryDirectory(prefix=f"{access_code}_", dir=tmpdir) as struct_tmpdir:
        sample_indices_list, template = frame_to_bioassembly_dict(traj, 0, access_code, struct_tmpdir)
        atom_array = template['atom_array']
        ligand_chain_id = atom_array.chain_id[-1]
        assert np.unique(atom_array.chain_id).shape[0] > 1, "Expected protein and ligand chains."

        sample_indices_list = [
            index for index in sample_indices_list
            if index['chain_2_id'] != ""
            and ligand_chain_id in (index['chain_1_id'], index['chain_2_id'])
        ]
        assert sample_indices_list, f"No first-frame interface includes ligand chain {ligand_chain_id}."

        atom_array_stack = [atom_array]
        for frame_no in range(1, traj.n_frames):
            _, frame_dict = frame_to_bioassembly_dict(traj, frame_no, access_code, struct_tmpdir)
            frame_atoms = frame_dict['atom_array']
            for annotation in ('chain_id', 'res_id', 'res_name', 'atom_name', 'element'):
                if not np.array_equal(getattr(atom_array, annotation), getattr(frame_atoms, annotation)):
                    raise ValueError(f"{struct} frame {frame_no}: atom identity changed ({annotation}).")
            atom_array_stack.append(frame_atoms)

    template['atom_array_stack'] = atom_array_stack
    save_path = os.path.join(outDir, access_code + '.pkl.gz')
    temporary_path = save_path + '.tmp'
    try:
        dump_gzip_pickle(template, temporary_path)
        os.replace(temporary_path, save_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

    return sample_indices_list


def main(args):
    os.makedirs(args.outDir, exist_ok=True)
    indices_path = args.outDir + ".csv"

    # make tmp dir
    tmpdir = os.path.join(args.outDir, "tmp")
    os.makedirs(tmpdir, exist_ok=True)

    # load struct list
    def txt_to_list(fpath: str):
        with open(fpath, "r", encoding="utf-8") as f:
            result_list = [line.strip() for line in f if line.strip()]

        return result_list

    structs = txt_to_list(args.splitFile)
    exclude_pdb_list = txt_to_list(args.excludeFile) if args.excludeFile else []
    structs = [struct for struct in structs if struct not in exclude_pdb_list]

    # process
    all_sample_indices_list = []
    failed = completed = 0
    started = time.time()
    failure_ids_path = os.path.join(args.outDir, "failed_pdb_ids.txt")
    failure_details_path = os.path.join(args.outDir, "failures.jsonl")

    def write_summary(status):
        summary = {
            "total": len(structs), "attempted": completed,
            "succeeded": completed - failed, "failed": failed,
            "num_workers": args.num_workers, "status": status,
            "elapsed_seconds": round(time.time() - started, 1),
            "index_rows": len(all_sample_indices_list),
        }
        temporary = Path(args.outDir) / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(Path(args.outDir) / "summary.json")
        return summary

    write_summary("running")
    try:
        with open(failure_ids_path, "w", encoding="utf-8") as failure_ids, \
                open(failure_details_path, "w", encoding="utf-8") as failure_details, \
                (Path(args.outDir) / "successes.jsonl").open("w", encoding="utf-8") as successes, \
                ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_one_struct, struct, args.trajDir, args.outDir, tmpdir): struct
                for struct in structs
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", mininterval=10):
                try:
                    sample_indices_list = future.result()
                except BrokenProcessPool:
                    raise
                except Exception as exc:
                    failed += 1
                    failure = {
                        "pdb_id": futures[future], "error_type": type(exc).__name__,
                        "message": str(exc), "traceback": traceback.format_exc(),
                    }
                    failure_ids.write(failure['pdb_id'] + '\n')
                    failure_details.write(json.dumps(failure) + '\n')
                    failure_ids.flush()
                    failure_details.flush()
                else:
                    all_sample_indices_list.extend(sample_indices_list)
                    successes.write(json.dumps({
                        "pdb_id": futures[future], "indices": sample_indices_list,
                    }, default=json_scalar) + "\n")
                    successes.flush()
                completed += 1
                write_summary("running")

        df = pd.DataFrame(all_sample_indices_list)
        if not df.empty:
            df = df.sort_values(['pdb_id', 'chain_1_id', 'chain_2_id'])
        temporary_indices = indices_path + ".tmp"
        df.to_csv(temporary_indices, index=False, quoting=csv.QUOTE_NONNUMERIC)
        os.replace(temporary_indices, indices_path)
    except BaseException:
        write_summary("interrupted")
        raise

    print(json.dumps(write_summary("completed")), flush=True)
    print(f"Failed PDB IDs: {failure_ids_path}")
    print(f"Failure details: {failure_details_path}")
    print(f"Interface indices: {indices_path}")


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--splitFile", required=True)
    parser.add_argument("-e", "--excludeFile", help="Optional file of PDB IDs to skip; no IDs are excluded by default.")
    parser.add_argument("-t", "--trajDir", required=True, type=str)
    parser.add_argument("-o", "--outDir", required=True, type=str)
    parser.add_argument("-n", "--num_workers", required=True, type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
