"""Assign MISATO chains from H5 metadata and write hydrogen-free PDB/XTC trajectories."""

import os
import pickle as pkl
import json
import tempfile
import time
import traceback
from pathlib import Path
import h5py
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import mdtraj as md
from tqdm import tqdm
atomic_numbers_Map = {1:'H', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F',11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',19:'K',20:'Ca',34:'Se',35:'Br',53:'I'}


def load_pkl(fpath):
    with open(fpath, "rb") as f:
        obj = pkl.load(f)

    return obj


def get_maps(mapdir):
    """
    Load the maps
    Args:
        mapdir: path to the maps
    """
    residueMap = load_pkl(os.path.join(mapdir, 'atoms_residue_map.pickle'))
    typeMap = load_pkl(os.path.join(mapdir, 'atoms_type_map.pickle'))
    nameMap = load_pkl(os.path.join(mapdir, 'atoms_name_map_for_pdb.pickle'))

    return residueMap, typeMap, nameMap


def update_residue_indices(
    residue_number, i, type_string, atoms_type, atoms_residue, residue_name, residue_atom_index,residue_Map, typeMap
):
    """
    Increment protein residue indices at O-N boundaries, accounting for
    side-chain O-N pairs in GLN and ASN. MOL retains one residue per chain.
    Args:
        residue_number: residue number
        i: atom index
        type_string: type of the atom
        atoms_type: type of the atoms
        atoms_residue: residue of the atoms
        residue_name: residue name
        residue_atom_index: atom index within the residue
        residue_Map: dictionary
        typeMap: dictionary
    """
    if i == len(atoms_type) - 1:
        # process ending, no need to update anything.
        return residue_number, residue_atom_index

    # First, i and i + 1 are in the same chain, otherwise we won't enter this method.
    # base on this, we can do:
    if residue_Map[atoms_residue[i]] == 'MOL':
        if residue_Map[atoms_residue[i + 1]] != 'MOL':
            raise ValueError("MOL res shouldn't exist in a protein chain")

        # Ligand's residue number only increase when chain ends.
        return residue_number, residue_atom_index

    else:
        if residue_Map[atoms_residue[i + 1]] == 'MOL':
            raise ValueError("MOL res shouldn't exist in a protein chain")

    # Only protein chain can arrive here.
    if type_string[0] == 'O' and typeMap[atoms_type[i+1]][0] == 'N':
        # GLN and ASN have a O-N sequence within the AA. See nameMap (atoms_name_map_for_pdb.pickle)
        if not (
            (residue_name == 'GLN' and residue_atom_index in [12, 14]) or \
                (residue_name == 'ASN' and residue_atom_index in [9, 11])
        ):
            residue_number += 1
            residue_atom_index = 0

    return residue_number, residue_atom_index


def get_seqres_info(
    atoms_type, atoms_number, atoms_residue,
    molecules_begin_atom_index,
    typeMap, residue_Map, nameMap
):
    seqres = [[]]

    residue_number = 1
    residue_atom_index = 0 # reset to 0 when entering the next residue.
    for i in range(len(atoms_type)):
        residue_atom_index += 1
        type_string = typeMap[atoms_type[i]]
        residue_name = residue_Map[atoms_residue[i]]

        if residue_atom_index == 1:
            seqres[-1].append(residue_name)

        if i + 1 in molecules_begin_atom_index: # entering the next chain
            residue_number += 1
            residue_atom_index = 0
            seqres.append([])

        else: # check whether entering the next residue
            residue_number, residue_atom_index = update_residue_indices(
                residue_number, i, type_string, atoms_type, atoms_residue, residue_name, residue_atom_index, residue_Map, typeMap
            )

    for i in range(len(seqres)):
        for j in range(len(seqres[i])):
            if seqres[i][j] == "HIE":
                seqres[i][j] = "HIS"
            if seqres[i][j] == "CYX":
                seqres[i][j] = "CYS"

    return seqres


def reassign_chain_ids(traj, chains_res_info):
    # Read residues and bonds from the trajectory topology.
    top = traj.topology
    residues = []
    for residue in top.residues:
        residues.append(residue)

    atoms, bonds = top.to_dataframe()

    # H5-derived chains must cover every trajectory residue.
    template_res_len = sum([len(chain_res) for chain_res in chains_res_info])
    assert template_res_len == len(residues), (
        f"Residue count mismatch: H5-derived chains contain {template_res_len} residues, "
        f"but the trajectory topology contains {len(residues)}."
    )

    # Build the topology with reassigned chain IDs.
    new_top = md.Topology()
    atom_index_map = {}

    # Assign residues to chains in the order supplied by the H5 data.
    resSeq_counter = 1
    for chain_idx, chain_res in enumerate(chains_res_info):
        new_chain = new_top.add_chain()

        res_counter = 0
        for residue in residues[resSeq_counter - 1:]:
            if res_counter >= len(chain_res):
                break

            res_name = residue.name

            # Residue identities must match the H5-derived chain sequence.
            assert res_name == chain_res[res_counter], f"{res_name} != {chain_res[res_counter]}"

            res = new_top.add_residue(res_name, new_chain, resSeq=resSeq_counter)

            for atom in residue.atoms:
                new_atom = new_top.add_atom(atom.name, atom.element, res)
                atom_index_map[atom.index] = new_atom

            res_counter += 1
            resSeq_counter += 1

    # Preserve bonds using the new atom indices.
    for bond in bonds:
        new_top.add_bond(atom_index_map[bond[0]], atom_index_map[bond[1]])

    # Retain the trajectory coordinates with the reassigned topology.
    new_traj = md.Trajectory(traj.xyz, new_top)
    return new_traj


def process_one_struct(struct, trajDir, datasetMD, mapDir, outDir):
    try:
        # Parse the AMBER topology explicitly for either supported suffix.
        topfile = os.path.join(trajDir, f"{struct}.prmtop")
        if not os.path.exists(topfile):
            topfile = os.path.join(trajDir, f"{struct}.top")

        traj_file = os.path.join(trajDir, f"{struct}.nc")
        traj = md.load(traj_file, top=md.load_prmtop(topfile))

        # prepare seqres_info
        with h5py.File(datasetMD, 'r') as h5:
            atoms_type = h5.get(struct+'/'+'atoms_type')[:]
            atoms_number = h5.get(struct+'/'+'atoms_number')[:]
            atoms_residue = h5.get(struct+'/'+'atoms_residue')[:]
            molecules_begin_atom_index = h5.get(struct+'/'+'molecules_begin_atom_index')[:]

        residueMap, typeMap, nameMap = get_maps(mapDir)

        seqres = get_seqres_info(
            atoms_type, atoms_number, atoms_residue, molecules_begin_atom_index,
            typeMap, residueMap, nameMap
        )

        # reassign_chain_ids
        traj = reassign_chain_ids(traj, seqres)

        # remove H
        traj.atom_slice([a.index for a in traj.top.atoms if a.element.symbol != 'H'], True)

        with tempfile.TemporaryDirectory(prefix=f".{struct}_", dir=outDir) as tmpdir:
            xtc = Path(tmpdir) / f"{struct}.xtc"
            pdb = Path(tmpdir) / f"{struct}.pdb"
            traj.save(str(xtc))
            traj[0].save_pdb(str(pdb))
            pdb.replace(Path(outDir) / pdb.name)
            xtc.replace(Path(outDir) / xtc.name)
    except Exception as exc:
        return {
            "pdb_id": struct,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }


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

    exclude_pdb_list = txt_to_list(args.excludeFile) if args.excludeFile else []
    structs = [struct for struct in structs if struct not in exclude_pdb_list]

    failure_ids_path = os.path.join(args.outDir, "failed_pdb_ids.txt")
    failure_details_path = os.path.join(args.outDir, "failures.jsonl")
    failed = completed = 0
    started = time.time()

    def write_summary(status):
        summary = {
            "total": len(structs), "attempted": completed,
            "succeeded": completed - failed, "failed": failed,
            "num_workers": args.num_workers, "status": status,
            "elapsed_seconds": round(time.time() - started, 1),
        }
        temporary = Path(args.outDir) / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(Path(args.outDir) / "summary.json")
        return summary

    write_summary("running")
    try:
        with open(failure_ids_path, "w", encoding="utf-8") as failure_ids, \
                open(failure_details_path, "w", encoding="utf-8") as failure_details, \
                (Path(args.outDir) / "successes.jsonl").open("w") as successes, \
                ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_one_struct, struct, args.trajDir, args.datasetMD, args.mapDir, args.outDir): struct
                for struct in structs
            }
            for future in tqdm(as_completed(futures), total=len(futures), mininterval=10):
                failure = future.result()
                if failure is not None:
                    failed += 1
                    failure_ids.write(failure["pdb_id"] + "\n")
                    failure_details.write(json.dumps(failure) + "\n")
                    failure_ids.flush()
                    failure_details.flush()
                else:
                    successes.write(json.dumps({"pdb_id": futures[future]}) + "\n")
                    successes.flush()
                completed += 1
                write_summary("running")
    except BaseException:
        write_summary("interrupted")
        raise

    print(json.dumps(write_summary("completed")), flush=True)
    print(f"Failed PDB IDs: {failure_ids_path}")
    print(f"Failure details: {failure_details_path}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Assign MISATO chains and remove hydrogen atoms. Progress, completed IDs, and failure details are written to the output directory."
    )
    parser.add_argument("-s", "--splitDir", required=True)
    parser.add_argument("-e", "--excludeFile", help="Optional file of PDB IDs to skip; no IDs are excluded by default.")
    parser.add_argument("-dMD", "--datasetMD", required=True, type=str)
    parser.add_argument("-t", "--trajDir", required=True, type=str,
                        help="Step 0 directory containing .nc coordinates and .top or .prmtop AMBER topologies.")
    parser.add_argument("-m", "--mapDir", required=True, type=str)
    parser.add_argument("-o", "--outDir", required=True, type=str)
    parser.add_argument("-n", "--num_workers", required=True, type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
