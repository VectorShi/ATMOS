"""Atom correspondence, C-alpha validity, and RMSD time-series error."""

import numpy as np
import torch


def align_tops(tops):
    names = []
    for top in tops:
        names.append([repr(a) for a in top.atoms])

    intersection = []
    for nam in names[0]:
        is_common = True
        for names_i in names:
            if nam not in names_i:
                is_common = False
                break

        if is_common: intersection.append(nam)

    masks = []
    for names_i in names:
        mask_i = [names_i.index(nam) for nam in intersection]
        masks.append(mask_i)

    return masks


def calculate_rmse_of_rmsd(pred_traj, gt_traj):
    pred_traj.atom_slice([a.index for a in pred_traj.top.atoms if a.name == 'CA'], True)
    gt_traj.atom_slice([a.index for a in gt_traj.top.atoms if a.name == 'CA'], True)

    pred_traj.superpose(pred_traj)
    gt_traj.superpose(gt_traj)

    def calculate_rmsd_seq(xyz):
        xyz = xyz * 10 # nm to A
        ref = xyz[0]  # (n_atoms, 3)
        diff = xyz - ref  # (n_frames, n_atoms, 3)
        sq_dist = np.sum(diff**2, axis=-1)
        rmsd = np.sqrt(np.mean(sq_dist, axis=-1))
        return rmsd

    pred_rmsd = calculate_rmsd_seq(pred_traj.xyz)
    traj_rmsd = calculate_rmsd_seq(gt_traj.xyz)

    def rmse(a, b):
        _len = a.shape[0]
        mse = np.sum((a - b) ** 2, axis=-1) / _len
        return mse ** 0.5

    return rmse(pred_rmsd, traj_rmsd)


def compute_validity(traj, clash_threshold=0.3, bond_break_threshold=0.419) -> object:
    alpha_carbons = traj.topology.select('name CA')
    alpha_carbons_xyz = traj.xyz[:, alpha_carbons, :]
    num_atoms = alpha_carbons_xyz.shape[1]
    alpha_carbons_xyz_torch = torch.from_numpy(alpha_carbons_xyz)
    distances = torch.cdist(alpha_carbons_xyz_torch, alpha_carbons_xyz_torch).numpy()
    # judge clash
    has_clash = np.sum(distances < clash_threshold, axis=(1, 2)) - num_atoms > 0
    # judge bond break
    adjacent_distances = distances[:, np.arange(num_atoms - 1), np.arange(1, num_atoms)]
    for i in range(1, num_atoms):
        if traj.topology.residue(i - 1).chain != traj.topology.residue(i).chain:
            adjacent_distances[:, i - 1] = -np.inf
    has_bond_break = np.sum(adjacent_distances > bond_break_threshold, axis=1) > 0
    valid_conformations = ~(has_clash | has_bond_break)
    val_ca = np.mean(valid_conformations)
    return val_ca, valid_conformations
