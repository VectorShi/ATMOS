"""Temporal RMSD, C-alpha validity, MSM transition RMSE, and TIC0 decay."""

import contextlib
import io
from pathlib import Path

import mdtraj as md
import numpy as np
from deeptime.decomposition import TICA
from statsmodels.tsa.stattools import acf

from atmos.evaluation.mdcath.geometry import (
    align_tops,
    calculate_rmse_of_rmsd,
    compute_validity,
)
from atmos.evaluation.mdcath.tica import (
    KMeansWrapper,
    get_kmeans,
    get_msm,
    msm_discretize,
    reduced_tica_features,
)


def transition_matrix_rmse(reference, prediction, nstates=10, lagtime=1):
    """Unweighted RMSE of row-normalized transition matrices, including empty rows."""
    def estimate(states):
        counts = np.zeros((nstates, nstates), dtype=float)
        if lagtime <= 0 or len(states) <= lagtime:
            return counts
        source, destination = states[:-lagtime], states[lagtime:]
        valid = (source >= 0) & (source < nstates) & (destination >= 0) & (destination < nstates)
        np.add.at(counts, (source[valid], destination[valid]), 1.0)
        totals = counts.sum(axis=1, keepdims=True)
        return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    return float(np.sqrt(np.mean((estimate(reference) - estimate(prediction)) ** 2)))


def evaluate_protein(protein, trajectory_dir, gt_dir, max_len):
    trajectory_dir, gt_dir = Path(trajectory_dir), Path(gt_dir)
    stem = f"mdcath_dataset_{protein}_0"
    prediction = md.load(str(trajectory_dir / f"{stem}.xtc"), top=str(trajectory_dir / f"{stem}.pdb"))
    references = [
        md.load(str(gt_dir / f"mdcath_dataset_{protein}_{replica}.xtc"), top=str(gt_dir / f"{stem}.pdb"))
        for replica in range(5)
    ]
    if prediction.n_frames != max_len or references[0].n_frames < max_len:
        raise ValueError(f"{protein}: expected {max_len} prediction frames and at least as many GT frames")

    # RMSD time series use C-alpha traces independently aligned to their own
    # first frames, before selecting common atoms for the dynamics features.
    temporal_rmse = calculate_rmse_of_rmsd(prediction[:], references[0][:max_len])
    ref_mask, pred_mask = align_tops([references[0].topology, prediction.topology])
    for reference in references:
        reference.atom_slice(ref_mask, inplace=True)
    prediction.atom_slice(pred_mask, inplace=True)

    reference_features = [reduced_tica_features(reference)[0] for reference in references]
    tica = TICA(lagtime=1, dim=2, scaling="kinetic_map").fit(reference_features).fetch_model()
    # Concatenate GT replicas for TIC0 autocorrelation and MSM transition
    # counting, including transitions across replica boundaries.
    joined = references[0]
    for replica in references[1:]:
        joined = joined.join(replica)
    ref_tics = tica.transform(reduced_tica_features(joined)[0])
    pred_tics = tica.transform(reduced_tica_features(prediction)[0])

    mean, std = ref_tics[:, 0].mean(), ref_tics[:, 0].std()
    if not np.isfinite(std) or std == 0:
        raise ValueError(f"{protein}: zero or non-finite reference TIC0 variance")
    ref_acf = acf((ref_tics[:max_len, 0] - mean) / std, nlags=100, adjusted=True)
    pred_acf = acf((pred_tics[:, 0] - mean) / std, nlags=100, adjusted=True)
    ref_crossings = np.flatnonzero(ref_acf < 0.5)
    pred_crossings = np.flatnonzero(pred_acf < 0.5)
    # An absent reference crossing cannot contribute to Pearson r. Predictions
    # use the trajectory length as the fallback when no crossing is found.
    ref_decay = float(ref_crossings[0]) if len(ref_crossings) else float("nan")
    pred_decay = float(pred_crossings[0]) if pred_acf[0] > 0.5 and len(pred_crossings) else float(max_len)

    validity, _ = compute_validity(prediction)
    kmeans, ref_microstates = get_kmeans(ref_tics, n_clusters=100, max_iter=100)
    with contextlib.redirect_stdout(io.StringIO()):
        msm, pcca = get_msm(ref_microstates, lagtime=1, nstates=10, n_clusters=100)
    wrapper = KMeansWrapper(kmeans, msm.count_model.state_symbols)
    pred_states = msm_discretize(pred_tics, wrapper, pcca)
    ref_states = msm_discretize(ref_tics, wrapper, pcca)
    return {
        "protein": protein,
        "rmse_of_rmsd": float(temporal_rmse),
        "val_ca": float(validity),
        "msm_tm_rmse": transition_matrix_rmse(ref_states, pred_states),
        "tic0_decorr_md": ref_decay,
        "tic0_decorr": pred_decay,
    }
