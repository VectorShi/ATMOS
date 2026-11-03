"""MISATO flexibility, Gaussian distribution, clash, and temporal metrics."""

import mdtraj as md
import numpy as np
from scipy.stats import pearsonr
import torch


POCKET_CUTOFF_ANGSTROM = 10.0
CLASH_CUTOFF_ANGSTROM = 1.1
# These IDs do not contribute to RMWD. Neither belongs to the fixed 100-target
# MISATO inclusion list, so this rule does not remove any benchmark target.
RMWD_EXCLUDED_TARGETS = {"3m3c", "6inz"}


def crop_pocket(traj, cutoff=POCKET_CUTOFF_ANGSTROM):
    """Select whole residues within cutoff of the ligand in reference frame 0.

    Input trajectories contain heavy atoms only. The final chain is the ligand;
    all preceding chains contain the protein. The selection is fixed over time.
    """
    chains = list(traj.top.chains)
    if len(chains) < 2:
        raise ValueError("A protein chain and a final ligand chain are required")
    ligand_indices = [atom.index for atom in chains[-1].atoms]
    protein_indices = [atom.index for chain in chains[:-1] for atom in chain.atoms]
    xyz = torch.as_tensor(traj.xyz[0] * 10)
    distances = torch.cdist(xyz[protein_indices], xyz[ligand_indices])
    row_for_atom = {index: row for row, index in enumerate(protein_indices)}
    pocket_indices = []
    for chain in chains[:-1]:
        for residue in chain.residues:
            atoms = [atom.index for atom in residue.atoms]
            if distances[[row_for_atom[index] for index in atoms]].min().item() <= cutoff:
                pocket_indices.extend(atoms)
    if not pocket_indices or not ligand_indices:
        raise ValueError("The reference must contain a nonempty pocket and ligand")
    pocket_indices = sorted(set(pocket_indices))
    return pocket_indices, ligand_indices, pocket_indices + ligand_indices


def calculate_rmsf(traj, indices):
    """Per-atom RMS fluctuation in Å, using the trajectory's own first frame."""
    traj = traj.atom_slice(indices, inplace=False)
    traj.superpose(traj)
    return md.rmsf(traj, traj[0]) * 10


def get_mean_covar(xyz):
    mean = xyz.mean(0)
    centered = xyz - mean
    covar = (centered[..., None] * centered[..., None, :]).mean(0)
    return mean, covar


def sqrtm(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return (eigenvectors * np.sqrt(eigenvalues[:, None])) @ np.linalg.inv(eigenvectors)


def calculate_emd_mean_var(pred_traj, gt_traj, indices):
    """Return RMWD mean/covariance components in Å and any fallback reason.

    Each trajectory is superposed onto its own first frame over the evaluated
    atoms. Covariances use population normalization (division by frame count).
    A numerical exception or complex covariance term replaces that term, for
    all atoms of this target and scope, with sqrt(trace(reference covariance)).
    """
    pred = pred_traj.atom_slice(indices, inplace=False)
    ref = gt_traj.atom_slice(indices, inplace=False)
    pred.superpose(pred)
    ref.superpose(ref)
    pred_mean, pred_covar = get_mean_covar(pred.xyz)
    ref_mean, ref_covar = get_mean_covar(ref.xyz)
    fallback = None
    with np.errstate(invalid="raise"):
        emd_mean = np.square(ref_mean - pred_mean).sum(-1) ** 0.5 * 10
        try:
            emd_var = np.trace(
                ref_covar + pred_covar - 2 * sqrtm(ref_covar @ pred_covar),
                axis1=1, axis2=2,
            ) ** 0.5 * 10
            if np.iscomplexobj(emd_var):
                raise ValueError("Complex covariance contribution")
        except Exception as exc:
            fallback = f"{type(exc).__name__}: {exc}"
            emd_var = np.trace(ref_covar, axis1=1, axis2=2) ** 0.5 * 10
        return np.square(emd_mean).mean() ** 0.5, np.square(emd_var).mean() ** 0.5, fallback


def calculate_intra_ligand_clashes(traj, indices):
    """Count unordered, nonbonded ligand atom pairs <=1.1 Å over all frames."""
    ligand = traj.atom_slice(indices, inplace=False)
    coordinates = torch.as_tensor(ligand.xyz) * 10
    distances = torch.cdist(coordinates, coordinates)
    excluded = torch.eye(ligand.n_atoms, dtype=torch.bool)
    for atom1, atom2 in ligand.top.bonds:
        excluded[atom1.index, atom2.index] = True
        excluded[atom2.index, atom1.index] = True
    clashes = distances <= CLASH_CUTOFF_ANGSTROM
    clashes[excluded.unsqueeze(0).expand_as(clashes)] = False
    return (clashes.float().sum() / 2).item()


def calculate_pocket_ligand_clashes(traj, pocket_indices, ligand_indices):
    """Count pocket–ligand atom pairs <=1.1 Å over all frames."""
    pocket = torch.as_tensor(traj.xyz[:, pocket_indices]) * 10
    ligand = torch.as_tensor(traj.xyz[:, ligand_indices]) * 10
    return (torch.cdist(pocket, ligand) <= CLASH_CUTOFF_ANGSTROM).float().sum().item()


def calculate_rmse_of_rmsd(pred_traj, gt_traj, indices):
    """RMSE in Å between RMSD-to-first-frame traces after separate alignment."""
    pred = pred_traj.atom_slice(indices, inplace=False)
    ref = gt_traj.atom_slice(indices, inplace=False)
    pred.superpose(pred)
    ref.superpose(ref)

    def trace(traj):
        xyz = traj.xyz * 10
        return np.sqrt(np.mean(np.sum((xyz - xyz[0]) ** 2, axis=-1), axis=-1))

    pred_rmsd, ref_rmsd = trace(pred), trace(ref)
    return (np.sum((pred_rmsd - ref_rmsd) ** 2) / len(pred_rmsd)) ** 0.5


def evaluate_trajectories(target, pred, ref):
    pocket, ligand, complex_atoms = crop_pocket(ref)
    rows = []
    for scope, indices in [("ligand", ligand), ("pocket_ligand", complex_atoms)]:
        pred_rmsf = calculate_rmsf(pred, indices)
        ref_rmsf = calculate_rmsf(ref, indices)
        correlation = pearsonr(pred_rmsf, ref_rmsf).statistic if len(indices) >= 2 else np.nan
        mean, var, fallback = np.nan, np.nan, None
        excluded = target in RMWD_EXCLUDED_TARGETS
        if not excluded:
            mean, var, fallback = calculate_emd_mean_var(pred, ref, indices)
        clash_count = (
            calculate_intra_ligand_clashes(pred, ligand) if scope == "ligand"
            else calculate_pocket_ligand_clashes(pred, pocket, ligand)
        )
        rows.append({
            "target": target, "scope": scope, "frames": pred.n_frames,
            "atoms": len(indices), "pocket_atoms": len(pocket), "ligand_atoms": len(ligand),
            "rmsf_pearson": float(correlation), "rmwd": float(np.sqrt(mean ** 2 + var ** 2)),
            "rmwd_mean_component": float(mean), "rmwd_covariance_component": float(var),
            "rmwd_excluded": excluded, "rmwd_covariance_fallback": fallback is not None,
            "rmwd_fallback_reason": fallback,
            "steric_clashes": clash_count,
            "t_rmsd_error": float(calculate_rmse_of_rmsd(pred, ref, indices)),
        })
    return rows
