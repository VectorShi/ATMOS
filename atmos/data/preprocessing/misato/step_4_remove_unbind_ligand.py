"""Check protein–ligand proximity using step 1 trajectories with MOL residues.

Progress, per-frame distances, passed IDs, and failures are recorded as each
complex finishes processing.
"""

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import mdtraj as md
from tqdm import tqdm
import torch


def process_one_struct(struct, trajDir):
    topfile = os.path.join(trajDir, f"{struct}.pdb")
    xtcfile = os.path.join(trajDir, f"{struct}.xtc")

    traj = md.load(xtcfile, top = topfile)

    prot_ca_traj = traj.atom_slice([a.index for a in traj.top.atoms if a.name == 'CA'], False)
    ligand_traj = traj.atom_slice([a.index for a in traj.top.atoms if a.residue.name == 'MOL'], False)

    prot_ca_traj = torch.as_tensor(prot_ca_traj.xyz * 10)
    ligand_traj = torch.as_tensor(ligand_traj.xyz * 10)
    if traj.n_frames != 100 or prot_ca_traj.shape[1] < 1 or ligand_traj.shape[1] < 1:
        raise ValueError(
            f"Expected 100 frames and nonempty CA and MOL selections; found "
            f"{traj.n_frames} frames, {prot_ca_traj.shape[1]} CA atoms, "
            f"{ligand_traj.shape[1]} MOL atoms."
        )
    if not torch.isfinite(prot_ca_traj).all() or not torch.isfinite(ligand_traj).all():
        raise ValueError("CA or MOL coordinates contain nonfinite values.")

    # Every frame must have a CA–MOL atom pair closer than 12 angstroms.
    # Distances use the input coordinates without periodic wrapping.
    distance_map = torch.cdist(prot_ca_traj, ligand_traj)
    prot_lig_distance_per_frame = torch.min(torch.min(distance_map, dim=-1).values, dim = -1).values
    maximum, frame = prot_lig_distance_per_frame.max(dim=0)
    return {
        "pdb_id": struct,
        "passed": bool(maximum < 12),
        "frames": traj.n_frames,
        "ca_atoms": prot_ca_traj.shape[1],
        "ligand_atoms": ligand_traj.shape[1],
        "max_min_distance_angstrom": float(maximum),
        "max_distance_frame_index": int(frame),
        "min_distance_per_frame_angstrom": prot_lig_distance_per_frame.tolist(),
    }


def main(args):
    output = Path(args.outDir)
    output.mkdir(parents=True, exist_ok=True)

    def txt_to_list(fpath: str):
        with open(fpath, "r", encoding="utf-8") as f:
            result_list = [line.strip() for line in f if line.strip()]

        return result_list

    splits = ["train_MD.txt", "val_MD.txt", "test_MD.txt"]
    structs = []
    for split_name in splits:
        split_file = os.path.join(args.splitDir, split_name)
        structs = structs + txt_to_list(split_file)

    exclude_pdb_list = txt_to_list(args.excludeFile) if args.excludeFile else []
    structs = [struct for struct in structs if struct not in exclude_pdb_list]

    results, failures = [], []
    rejected = errors = completed = 0
    started = time.time()

    def write_summary(status):
        summary = {
            "total": len(structs), "attempted": completed,
            "succeeded": len(results) - rejected, "failed": len(failures),
            "distance_rejected": rejected, "processing_errors": errors,
            "num_workers": args.num_workers, "status": status,
            "elapsed_seconds": round(time.time() - started, 1),
            "distance_threshold_angstrom": 12.0,
            "criterion": "All 100 frames must have minimum CA–MOL distance < 12 angstroms.",
        }
        temporary = output / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(output / "summary.json")
        return summary

    write_summary("running")
    try:
        with (output / "distances.jsonl").open("w") as measurements, \
                (output / "failures.jsonl").open("w") as failure_log, \
                (output / "failed_pdb_ids.txt").open("w") as failure_ids, \
                (output / "passed_pdb_ids.txt").open("w") as passed_ids, \
                ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_one_struct, struct, args.trajDir): struct for struct in structs}
            for future in tqdm(as_completed(futures), total=len(futures), mininterval=10):
                failure = None
                try:
                    result = future.result()
                except BrokenProcessPool:
                    raise
                except Exception as exc:
                    errors += 1
                    failure = {
                        "pdb_id": futures[future], "error_type": type(exc).__name__,
                        "message": str(exc), "traceback": traceback.format_exc(),
                    }
                else:
                    results.append(result)
                    measurements.write(json.dumps(result) + "\n")
                    measurements.flush()
                    if result["passed"]:
                        passed_ids.write(result["pdb_id"] + "\n")
                        passed_ids.flush()
                    else:
                        rejected += 1
                        failure = {
                            "pdb_id": result["pdb_id"],
                            "error_type": "LigandDistanceThresholdExceeded",
                            "message": "Maximum over frames of the minimum CA–MOL distance is >= 12 angstroms.",
                            "max_min_distance_angstrom": result["max_min_distance_angstrom"],
                            "max_distance_frame_index": result["max_distance_frame_index"],
                        }
                if failure is not None:
                    failures.append(failure)
                    failure_log.write(json.dumps(failure) + "\n")
                    failure_ids.write(failure["pdb_id"] + "\n")
                    failure_log.flush()
                    failure_ids.flush()
                completed += 1
                write_summary("running")

        results.sort(key=lambda item: item["pdb_id"])
        failures.sort(key=lambda item: item["pdb_id"])
        for name, content in {
            "distances.jsonl": "".join(json.dumps(item) + "\n" for item in results),
            "failures.jsonl": "".join(json.dumps(item) + "\n" for item in failures),
            "failed_pdb_ids.txt": "".join(item["pdb_id"] + "\n" for item in failures),
            "passed_pdb_ids.txt": "".join(item["pdb_id"] + "\n" for item in results if item["passed"]),
        }.items():
            temporary = output / (name + ".tmp")
            temporary.write_text(content)
            temporary.replace(output / name)
    except BaseException:
        write_summary("interrupted")
        raise

    print(json.dumps(write_summary("completed")), flush=True)
    for failure in failures:
        print(f"{failure['pdb_id']}: {failure['message']}", flush=True)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--splitDir", required=True, help="Directory containing train/val/test_MD.txt.")
    parser.add_argument("-e", "--excludeFile", help="Optional PDB IDs to skip, one per line.")
    parser.add_argument("-t", "--trajDir", required=True, help="Step 1 PDB/XTC directory, with ligand residues named MOL.")
    parser.add_argument("-o", "--outDir", required=True, help="Directory for distance measurements, exclusions, and summary.")
    parser.add_argument("-n", "--num_workers", required=True, type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
