"""Assign ligand residue and atom names from PDBbind MOL2 files."""

import os
import argparse
import json
import tempfile
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import mdtraj as md
from tqdm import tqdm


def parse_mol2(mol2_path):
    with open(mol2_path, 'r') as f:
        lines = f.readlines()

    start_line_idx = 0
    end_line_idx = 0
    for i, line in enumerate(lines):
        if "@<TRIPOS>ATOM" in line:
            start_line_idx = i + 1

        if start_line_idx and i >= start_line_idx and "@<TRIPOS>" in line:
            end_line_idx = i
            break

    assert start_line_idx and end_line_idx and end_line_idx - start_line_idx, (
        f"No complete, nonempty MOL2 atom section in {mol2_path}"
    )

    atom_lines = lines[start_line_idx:end_line_idx]
    atoms_mol2 = [line.strip().split() for line in atom_lines]

    for i, line in enumerate(atoms_mol2):
        assert len(line) == 9, (
            f"Expected 9 MOL2 atom fields in {mol2_path} at line {start_line_idx + i + 1}, "
            f"found {len(line)}"
        )
        if '.' in line[5]: # atom type
            line[5] = line[5].split('.')[0]

        atoms_mol2[i] = [line[1], line[5], line[7]] # [atom_name, atom_type, CCD]

    atoms_mol2 = [line for line in atoms_mol2 if line[1] != "H"] # remove H
    assert len(atoms_mol2), f"No heavy atoms in {mol2_path}"

    return atoms_mol2


def split_res_in_mol(atoms_mol2):
    residues_mol2 = [{"CCD": atoms_mol2[0][2], "atoms": []}]
    cached_atom_names = []

    for i in range(len(atoms_mol2)):
        atom_name, _, ccd = atoms_mol2[i]
        if atom_name not in cached_atom_names and ccd == residues_mol2[-1]["CCD"]:
            residues_mol2[-1]["atoms"].append(atoms_mol2[i])
            cached_atom_names.append(atom_name)

        else:
            residues_mol2.append(
                {"CCD": ccd, "atoms": [atoms_mol2[i]]}
            )
            cached_atom_names.clear()
            cached_atom_names.append(atom_name)

    return residues_mol2


def traj_mol_check(traj):
    top = traj.topology
    residues_names = []
    for residue in top.residues:
        residues_names.append(residue.name)

    MOL_IDXs = []
    for i, res_name in enumerate(residues_names):
        if res_name == "MOL":
            MOL_IDXs.append(i)

    assert len(MOL_IDXs) == 1 and MOL_IDXs[0] == len(residues_names) - 1, (
        f"Expected one MOL residue at the end of the trajectory topology; "
        f"found {len(MOL_IDXs)} at residue indices {MOL_IDXs} "
        f"among {len(residues_names)} residues."
    )


def traj_mol2_compare(traj, atoms_mol2):
    top = traj.topology
    mol_residues = []
    for residue in top.residues:
        if residue.name == "MOL":
            mol_residues.append(residue)

    assert len(mol_residues) == 1, f"Expected one MOL residue, found {len(mol_residues)}"
    mol_residue = mol_residues[0]

    mol_elements = []
    for atom in mol_residue.atoms:
        mol_elements.append(atom.element.symbol)

    mol_elements = tuple(mol_elements)

    mol_elements_mol2 = []
    for line in atoms_mol2:
        mol_elements_mol2.append(line[1])

    mol_elements_mol2 = tuple(mol_elements_mol2)

    if mol_elements != mol_elements_mol2:
        first_mismatch = next(
            (i for i, (actual, expected) in enumerate(zip(mol_elements, mol_elements_mol2)) if actual != expected),
            min(len(mol_elements), len(mol_elements_mol2)),
        )
        raise ValueError(
            f"Ligand heavy-atom element sequence differs: trajectory has {len(mol_elements)} atoms, "
            f"MOL2 has {len(mol_elements_mol2)} atoms; first difference at index {first_mismatch}."
        )


def update_mol_res_name_atom_name(traj, atoms_mol2):
    # The final chain must contain the single MOL residue. Matching heavy-atom
    # element sequences are required; atom names are assigned in MOL2 order.

    top = traj.topology

    # new topology
    new_top = md.Topology()
    atom_map = {}

    # add prot chains
    for chain in top.chains:
        chain_is_MOL = False
        for residue in chain.residues:
            if residue.name == "MOL":
                chain_is_MOL = True
            break

        if chain_is_MOL:
            break

        new_chain = new_top.add_chain(chain.chain_id)
        for residue in chain.residues:
            new_res = new_top.add_residue(residue.name, new_chain, resSeq=residue.resSeq)

            for atom in residue.atoms:
                new_atom = new_top.add_atom(atom.name, atom.element, new_res)
                atom_map[atom.index] = new_atom

    # update mol chain
    residues_mol2 = split_res_in_mol(atoms_mol2)

    mol_chain, mol_res = None, None
    for chain in top.chains:
        chain_is_MOL = False
        for residue in chain.residues:
            if residue.name == "MOL":
                chain_is_MOL = True
            break

        if not chain_is_MOL: continue

        mol_chain = chain
        for residue in mol_chain.residues:
            mol_res = residue
            break

    assert mol_chain is not None and mol_res is not None, "MOL must begin its own chain."
    new_chain = new_top.add_chain(mol_chain.chain_id)

    mol_res_Seq = mol_res.resSeq
    atoms_iter = mol_res.atoms
    for res_mol2 in residues_mol2:
        new_res = new_top.add_residue(res_mol2['CCD'], new_chain, resSeq=mol_res_Seq)
        mol_res_Seq = mol_res_Seq + 1

        for atom_line in res_mol2['atoms']:
            atom_name = atom_line[0]
            atom = next(atoms_iter)
            new_atom = new_top.add_atom(atom_name, atom.element, new_res)
            atom_map[atom.index] = new_atom


    for bond in top.bonds:
        new_top.add_bond(
            atom_map[bond[0].index],
            atom_map[bond[1].index]
        )

    new_traj = md.Trajectory(traj.xyz, new_top)

    return new_traj


def process_one_struct(struct, trajDir, pdbbindMol2, pdbbindMol2_alt, outDir):
    # parse mol2
    mol2_path = os.path.join(pdbbindMol2, struct.lower(), f"{struct.lower()}_ligand.mol2")
    mol2_path_alt = os.path.join(pdbbindMol2_alt, f"{struct.lower()}_ligand.mol2")
    assert os.path.exists(mol2_path) or os.path.exists(mol2_path_alt), (
        f"Missing ligand MOL2: tried {mol2_path} and {mol2_path_alt}"
    )
    if not os.path.exists(mol2_path):
        mol2_path = mol2_path_alt

    atoms_mol2 = parse_mol2(mol2_path)

    # load traj
    topfile = os.path.join(trajDir, f"{struct}.pdb")
    xtcfile = os.path.join(trajDir, f"{struct}.xtc")
    assert os.path.exists(topfile) and os.path.exists(xtcfile), (
        f"Missing trajectory input: expected {topfile} and {xtcfile}"
    )

    traj = md.load(xtcfile, top = topfile)
    traj_mol_check(traj)
    traj_mol2_compare(traj, atoms_mol2)

    # update_mol_res_name_atom_name, to match the MOL2 file.
    traj = update_mol_res_name_atom_name(traj, atoms_mol2)

    with tempfile.TemporaryDirectory(prefix=f".{struct}_", dir=outDir) as tmpdir:
        xtc = Path(tmpdir) / f"{struct}.xtc"
        pdb = Path(tmpdir) / f"{struct}.pdb"
        traj.save(str(xtc))
        traj[0].save_pdb(str(pdb))
        pdb.replace(Path(outDir) / pdb.name)
        xtc.replace(Path(outDir) / xtc.name)


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
                executor.submit(process_one_struct, struct, args.trajDir, args.pdbbindMol2, args.pdbbindMol2_alt, args.outDir): struct
                for struct in structs
            }
            for future in tqdm(as_completed(futures), total=len(futures), mininterval=10):
                try:
                    future.result()
                    successes.write(json.dumps({"pdb_id": futures[future]}) + "\n")
                    successes.flush()
                except BrokenProcessPool:
                    raise
                except Exception as exc:
                    failed += 1
                    failure = {
                        "pdb_id": futures[future], "error_type": type(exc).__name__,
                        "message": str(exc), "traceback": traceback.format_exc(),
                    }
                    failure_ids.write(failure["pdb_id"] + "\n")
                    failure_details.write(json.dumps(failure) + "\n")
                    failure_ids.flush()
                    failure_details.flush()
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
        description="Assign ligand residue and atom names from PDBbind MOL2 files. Progress, completed IDs, and failure details are written to the output directory."
    )
    parser.add_argument("-s", "--splitDir", required=True)
    parser.add_argument("-e", "--excludeFile", help="Optional file of PDB IDs to skip; no IDs are excluded by default.")
    parser.add_argument("-t", "--trajDir", required=True, type=str)
    parser.add_argument("-p", "--pdbbindMol2", required=True, type=str,
                        help="MOL2 root containing <lowercase_id>/<lowercase_id>_ligand.mol2.")
    parser.add_argument("-p1", "--pdbbindMol2_alt", required=True, type=str,
                        help="Fallback directory containing <lowercase_id>_ligand.mol2; used only when the primary file is absent.")
    parser.add_argument("-o", "--outDir", required=True, type=str)
    parser.add_argument("-n", "--num_workers", required=True, type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
