"""Read matched MISATO trajectories and restore reference ligand connectivity."""

from copy import deepcopy
import gzip
from pathlib import Path
import pickle
import re
from tempfile import TemporaryDirectory

import mdtraj as md
import numpy as np


FRAME_COUNT = 100


def read_ids(path):
    return {line.strip().lower() for line in Path(path).read_text().splitlines() if line.strip()}


def file_index(directory, suffix):
    result = {}
    for path in sorted(Path(directory).glob(f"*{suffix}")):
        target = path.name[:-len(suffix)].lower()
        if target in result:
            raise ValueError(f"Duplicate file for {target} in {directory}")
        result[target] = path.resolve()
    return result


def discover_inputs(work_dirs, target_ids, gt_dir=None):
    """Collect one prediction per target from runs or parallel-launch directories."""
    tasks = {}
    for work in work_dirs:
        work = work.resolve()
        if (work / "predictions").is_dir() or (work / "trajectories").is_dir():
            runs = [work]
        else:
            runs = sorted(path.parent for path in work.glob("*/predictions") if path.is_dir())
        if not runs:
            raise FileNotFoundError(f"No predictions/ or trajectories/ found under {work}")
        for run in runs:
            if (run / "predictions").is_dir():
                entries = []
                for path in sorted((run / "predictions").glob("*.pkl.gz")):
                    match = re.fullmatch(r"rank_\d+_(.+)\.pkl\.gz", path.name)
                    if match:
                        entries.append((match[1].lower(), {"prediction_pickle": str(path.resolve())}))
            else:
                pred_pdb = file_index(run / "trajectories", ".pdb")
                ref = gt_dir or run / "gt_trajectories"
                ref_xtc, ref_pdb = file_index(ref, ".xtc"), file_index(ref, ".pdb")
                entries = []
                for target, path in file_index(run / "trajectories", ".xtc").items():
                    if target not in target_ids:
                        continue
                    if target not in pred_pdb or target not in ref_xtc or target not in ref_pdb:
                        raise FileNotFoundError(f"{target}: prediction and reference each need PDB and XTC")
                    entries.append((target, {"pred_xtc": str(path), "pred_pdb": str(pred_pdb[target]),
                                            "ref_xtc": str(ref_xtc[target]), "ref_pdb": str(ref_pdb[target])}))
            for target, source in entries:
                if target not in target_ids:
                    continue
                if target in tasks:
                    raise ValueError(f"Duplicate prediction for {target}; input groups must be disjoint")
                tasks[target] = {"target": target, **source}
    if not tasks:
        raise ValueError("No predictions match the MISATO target list")
    if gt_dir is not None and any("prediction_pickle" in task for task in tasks.values()):
        raise ValueError("--gt-dir applies to PDB/XTC inputs; prediction pickles already contain reference coordinates")
    return [tasks[target] for target in sorted(tasks)]


def load_pickle(path):
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def first_atom_array(data):
    frames = data.get("atom_array_stack", data.get("atom_array"))
    if frames is None:
        raise ValueError("Reference pickle has no atom_array_stack or atom_array")
    return frames[0]


def convert_coordinates(atom_array, coordinates, output_dir, target):
    """Write PDB/XTC after sequential rigid alignment and PDB coordinate rounding."""
    import biotite.structure.io.pdb as pdb
    from protenix.metrics.rmsd import align_pred_to_true

    output_dir.mkdir(parents=True, exist_ok=True)
    atom_array = deepcopy(atom_array)
    with TemporaryDirectory(prefix=f"{target}_", dir=output_dir) as temporary:
        pdb_paths = []
        last = None
        for frame, coordinate in enumerate(coordinates):
            if last is not None:
                coordinate = align_pred_to_true(coordinate, last)[0]
            last = coordinate
            atom_array.coord = coordinate.cpu().numpy()
            path = Path(temporary) / f"{frame:03d}.pdb"
            pdb_file = pdb.PDBFile()
            pdb.set_structure(pdb_file, atom_array)
            pdb_file.write(path)
            pdb_paths.append(str(path))
        traj = md.load(pdb_paths)
        traj.save_xtc(str(output_dir / f"{target}.xtc"))
        traj[0].save_pdb(str(output_dir / f"{target}.pdb"))
    # Evaluate the saved files, including XTC quantization, just as external inputs.
    return md.load(str(output_dir / f"{target}.xtc"), top=str(output_dir / f"{target}.pdb"))


def heavy_atoms(traj):
    if any(atom.element is None for atom in traj.top.atoms):
        raise ValueError("Every atom must have an element for heavy-atom selection")
    return traj.atom_slice([atom.index for atom in traj.top.atoms if atom.element.symbol not in {"H", "D"}])


def topology_signature(traj):
    return [(atom.residue.chain.index, atom.residue.resSeq, atom.residue.name, atom.name, atom.element.symbol)
            for atom in traj.top.atoms]


def add_ligand_bonds(traj, reference_atoms):
    """Transfer direct ligand bonds by (residue name, atom name).

    Repeated keys resolve to their final atom in topology order, including
    ligands with multiple residues of the same name.
    """
    ligand = list(traj.top.chains)[-1]
    atoms = {(atom.residue.name, atom.name): atom for atom in ligand.atoms}
    if reference_atoms.bonds is None:
        raise ValueError("The bond reference has no bond table")
    chain = reference_atoms.chain_id[-1]
    reference_names = {(str(atom.res_name), str(atom.atom_name))
                       for atom in reference_atoms if atom.chain_id == chain}
    if set(atoms) - reference_names:
        raise ValueError(f"Ligand atoms missing from the bond reference: {sorted(set(atoms) - reference_names)}")
    existing = {tuple(sorted((a.index, b.index))) for a, b in traj.top.bonds}
    for left, right, _ in reference_atoms.bonds.as_array():
        a, b = reference_atoms[int(left)], reference_atoms[int(right)]
        if a.chain_id != chain or b.chain_id != chain:
            continue
        key_a, key_b = (str(a.res_name), str(a.atom_name)), (str(b.res_name), str(b.atom_name))
        if key_a in atoms and key_b in atoms:
            pair = tuple(sorted((atoms[key_a].index, atoms[key_b].index)))
            if pair not in existing:
                traj.top.add_bond(atoms[key_a], atoms[key_b])
                existing.add(pair)


def load_trajectories(task, output_dir, bond_reference_dir=None):
    target = task["target"]
    if "prediction_pickle" in task:
        import torch

        batch = load_pickle(task["prediction_pickle"])
        if str(batch["basic"]["pdb_id"]).lower() != target:
            raise ValueError("Prediction filename and stored target ID disagree")
        atom_array = batch.get("cropped_atom_array", batch.get("atom_array"))
        mask = batch["input_feature_dict"]["coordinate_mask"][0, 0].bool().cpu()
        reference_atoms = atom_array[mask.numpy()]
        trajectories = []
        for key, name in [("pred_dict", "trajectories"), ("label_dict", "gt_trajectories")]:
            coordinates = batch[key]["coordinate"].detach().cpu()
            if coordinates.ndim != 4 or coordinates.shape[0] != 1 or coordinates.shape[1] < FRAME_COUNT:
                raise ValueError(f"{key}: expected one trajectory with at least {FRAME_COUNT} frames")
            if coordinates.shape[2:] != (len(atom_array), 3):
                raise ValueError(f"{key}: coordinate shape does not match the atom array")
            coordinates = coordinates[0, :FRAME_COUNT, mask, :]
            if not torch.isfinite(coordinates).all():
                raise ValueError(f"{key}: nonfinite coordinates")
            trajectories.append(convert_coordinates(reference_atoms, coordinates, output_dir / name, target))
        pred, ref = trajectories
    else:
        pred = md.load(task["pred_xtc"], top=task["pred_pdb"])
        ref = md.load(task["ref_xtc"], top=task["ref_pdb"])
        if bond_reference_dir is None:
            raise ValueError("PDB/XTC inputs require --bond-reference-dir with MISATO bioassembly pickles")
        reference_path = Path(bond_reference_dir) / f"{target}.pkl.gz"
        reference_atoms = first_atom_array(load_pickle(reference_path))

    pred, ref = heavy_atoms(pred), heavy_atoms(ref)
    for traj in (pred, ref):
        if traj.n_frames != FRAME_COUNT:
            raise ValueError(f"Expected {FRAME_COUNT} frames, found {traj.n_frames}")
        if not np.isfinite(traj.xyz).all():
            raise ValueError("Nonfinite trajectory coordinates")
        if traj.n_chains < 2:
            raise ValueError("A protein chain and a final ligand chain are required")
    if topology_signature(pred) != topology_signature(ref):
        raise ValueError("Prediction/reference heavy-atom identities or ordering differ")
    add_ligand_bonds(pred, reference_atoms)
    add_ligand_bonds(ref, reference_atoms)
    return pred, ref
