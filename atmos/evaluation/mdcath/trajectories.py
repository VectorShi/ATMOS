"""Convert mdCATH predictions into PDB topologies and XTC trajectories."""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import biotite.structure.io.pdb as pdb
import copy
import os
import mdtraj as md
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse
from protenix.utils.file_io import load_gzip_pickle
from protenix.metrics.rmsd import align_pred_to_true


EXCLUDED_MDCATH_IDS = {"2noga01", "1waae00"}


def should_skip_trajectory(raw_traj_file):
    return any(excluded_id in raw_traj_file.lower() for excluded_id in EXCLUDED_MDCATH_IDS)


def remove_skipped_outputs(raw_traj_file, trajectory_dir):
    fname = os.path.basename(raw_traj_file).split('.')[0]
    pid_rid = '_'.join(fname.split('_')[2:])
    for ext in ("pdb", "xtc"):
        output_path = os.path.join(trajectory_dir, f"{pid_rid}.{ext}")
        if os.path.exists(output_path):
            os.unlink(output_path)


def pred_coord_to_pdb(
    atom_array,
    pred_coordinate,
    save_file
):
    pdb_file = pdb.PDBFile()
    pred_atom_array = copy.deepcopy(atom_array)
    pred_atom_array.coord = pred_coordinate.cpu().numpy()
    pdb.set_structure(pdb_file, pred_atom_array)
    pdb_file.write(save_file)


def process_one_generated_traj(
    raw_traj_file, trajectory_dir, tmpdir, max_len = 100, gt = False
):
    fname = os.path.basename(raw_traj_file).split('.')[0]
    pid_rid = '_'.join(fname.split('_')[2:]) # e.g. "rank_7_mdcath_dataset_1wr8a02_3.pkl.gz"

    batch = load_gzip_pickle(raw_traj_file)
    if not gt:
        structures = batch['pred_dict']['coordinate']
        structures = structures[:, :max_len] # [1, N_frame, N_atom, 3]
    else:
        structures = batch['label_dict']['coordinate']

    coordinate_mask = batch['input_feature_dict']['coordinate_mask'][0, 0] # [N_atom]
    coordinate_mask_bool = coordinate_mask.bool()
    structures = structures[0, :, coordinate_mask_bool, :]

    atom_array = batch['cropped_atom_array'] if 'cropped_atom_array' in batch.keys() else batch['atom_array']
    mask_indices = coordinate_mask_bool.cpu().numpy()
    filtered_atom_array = atom_array[mask_indices]

    save_files = []
    last_coordinate = None
    for frame_no in range(structures.shape[0]):
        pred_coordinate = structures[frame_no]

        if last_coordinate is not None:
            pred_coordinate = align_pred_to_true(
                pred_coordinate,
                last_coordinate,
            )[0]
        last_coordinate = pred_coordinate

        save_file = os.path.join(tmpdir, f"{pid_rid}_{frame_no:03d}.pdb")
        pred_coord_to_pdb(
            filtered_atom_array, pred_coordinate, save_file
        )
        save_files.append(save_file)

    traj = md.load(save_files)
    traj.save(os.path.join(trajectory_dir, f"{pid_rid}.xtc"))
    traj[0].save_pdb(os.path.join(trajectory_dir, f"{pid_rid}.pdb"))

    for save_file in save_files:
        os.unlink(save_file)


def run_task(args):
    fpath, trajectory_dir, tmp_dir, max_len, gt = args
    try:
        process_one_generated_traj(fpath, trajectory_dir=trajectory_dir, tmpdir=tmp_dir, max_len=max_len, gt=gt)
    except Exception as e:
        print(fpath, e)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--work_dir",
        type=str,
        required=True,
        help="Path to the inference working directory.",
    )
    argument_parser.add_argument(
        "--max_len",
        type=int,
        required=True,
        help="Maximum length for one processed trajectory.",
    )
    argument_parser.add_argument(
        "--gt",
        action="store_true",
        help="Use ground truth data for processing."
    )
    args = argument_parser.parse_args()

    prediction_dir = os.path.join(args.work_dir, "predictions")
    if not args.gt:
        trajectory_dir = os.path.join(args.work_dir, "trajectories")
    else:
        trajectory_dir = os.path.join(args.work_dir, "gt_trajectories")
    tmp_dir = os.path.join(args.work_dir, "tmp")

    if not os.path.exists(prediction_dir): return
    os.makedirs(trajectory_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    tasks = []
    for fname in os.listdir(prediction_dir):
        fpath = os.path.join(prediction_dir, fname)
        if should_skip_trajectory(fpath):
            remove_skipped_outputs(fpath, trajectory_dir)
            continue
        tasks.append((fpath, trajectory_dir, tmp_dir, args.max_len, args.gt))

    with ProcessPoolExecutor(max_workers=32) as executor:
        list(tqdm(executor.map(run_task, tasks), total=len(tasks)))


if __name__ == "__main__":
    main()
