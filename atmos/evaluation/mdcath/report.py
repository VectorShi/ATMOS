"""Aggregate mdCATH metrics across proteins and residues."""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def ensemble_per_protein(data):
    rows = []
    for name, out in data.items():
        crystal_contacts = out["crystal_distmat"] < 0.8
        ref_transient = (~crystal_contacts) & (out["ref_contact_prob"] > 0.1)
        pred_transient = (~crystal_contacts) & (out["af_contact_prob"] > 0.1)
        ref_weak = crystal_contacts & (out["ref_contact_prob"] < 0.9)
        pred_weak = crystal_contacts & (out["af_contact_prob"] < 0.9)
        buried = out["crystal_sasa"][0] < 0.02
        ref_exposed = (out["ref_sa_prob"] > 0.1) & buried
        pred_exposed = (out["af_sa_prob"] > 0.1) & buried
        rows.append({
            "protein": name,
            "md_pairwise": out["ref_mean_pairwise_rmsd"],
            "pred_pairwise": out["af_mean_pairwise_rmsd"],
            "cosine_sim": abs(out["cosine_sim"]),
            "emd_mean": np.square(out["emd_mean"]).mean() ** 0.5,
            "emd_var": np.square(out["emd_var"]).mean() ** 0.5,
            "rmsf_pearson": pearsonr(out["af_rmsf"], out["ref_rmsf"])[0],
            "md_pca_w2": out["EMD,ref"]["ref|af"],
            "joint_pca_w2": out["EMD,joint"]["ref|af"],
            "weak_contacts_j": (ref_weak & pred_weak).sum() / (ref_weak | pred_weak).sum(),
            "transient_contacts_j": (ref_transient & pred_transient).sum() / (ref_transient | pred_transient).sum(),
            "exposed_residue_j": (ref_exposed & pred_exposed).sum() / (ref_exposed | pred_exposed).sum(),
            "exposed_mi_rho": spearmanr(out["ref_mi_mat"].flatten(), out["af_mi_mat"].flatten())[0],
        })
    return pd.DataFrame(rows).set_index("protein")


def build_report(ensemble_data, dynamics):
    # Undefined sidechain statistics in CA-only mode remain NaN and carry n=0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _build_report(ensemble_data, dynamics)


def _build_report(ensemble_data, dynamics):
    per_protein = ensemble_per_protein(ensemble_data)
    if set(per_protein.index) != set(dynamics.index):
        raise ValueError("Ensemble and dynamics results do not cover the same proteins")
    per_protein = per_protein.join(dynamics).sort_index()
    count = len(per_protein)
    ref_rmsf = np.concatenate([ensemble_data[name]["ref_rmsf"] for name in per_protein.index])
    pred_rmsf = np.concatenate([ensemble_data[name]["af_rmsf"] for name in per_protein.index])
    rows = []

    def add(metric, value, aggregation, n=count, std=np.nan, unit=""):
        rows.append({"metric": metric, "aggregation": aggregation, "value": float(value),
                     "std": float(std), "n": int(n), "unit": unit})

    def median(metric, values, unit=""):
        add(metric, values.median(), "median over proteins", n=values.notna().sum(), unit=unit)

    median("MD pairwise RMSD", per_protein.md_pairwise, "angstrom")
    median("Pairwise RMSD", per_protein.pred_pairwise, "angstrom")
    add("Pairwise RMSD r", pearsonr(per_protein.md_pairwise, per_protein.pred_pairwise)[0], "Pearson r")
    add("MD RMSF", np.median(ref_rmsf), "median over pooled residues", unit="angstrom")
    add("RMSF", np.median(pred_rmsf), "median over pooled residues", unit="angstrom")
    add("Global RMSF r", pearsonr(ref_rmsf, pred_rmsf)[0], "Pearson r over pooled residues")
    median("Per target RMSF r", per_protein.rmsf_pearson)
    median("RMWD", np.sqrt(per_protein.emd_mean ** 2 + per_protein.emd_var ** 2), "angstrom")
    median("RMWD trans", per_protein.emd_mean, "angstrom")
    median("RMWD var", per_protein.emd_var, "angstrom")
    median("MD PCA W2", per_protein.md_pca_w2, "angstrom")
    median("Joint PCA W2", per_protein.joint_pca_w2, "angstrom")
    add("PC sim > 0.5 %", (per_protein.cosine_sim > 0.5).mean() * 100, "fraction times 100", unit="percent")
    median("Weak contacts J", per_protein.weak_contacts_j)
    add("Weak contacts nans", per_protein.weak_contacts_j.isna().mean(), "NaN fraction")
    median("Transient contacts J", per_protein.transient_contacts_j)
    add("Transient contacts nans", per_protein.transient_contacts_j.isna().mean(), "NaN fraction")
    median("Exposed residue J", per_protein.exposed_residue_j)
    median("Exposed MI matrix rho", per_protein.exposed_mi_rho)
    add("rmse_of_rmsd", per_protein.rmse_of_rmsd.mean(), "mean over proteins", unit="angstrom")
    for metric in ("val_ca", "msm_tm_rmse"):
        values = per_protein[metric]
        add(metric, values.mean(), "mean over proteins", n=values.notna().sum(), std=values.std())
    pairs = per_protein[["tic0_decorr", "tic0_decorr_md"]].replace([np.inf, -np.inf], np.nan).dropna()
    pearson = pearsonr(pairs.tic0_decorr, pairs.tic0_decorr_md)[0] if len(pairs) >= 2 else np.nan
    add("TIC0 decorrelation Pearson r", pearson, "Pearson r", n=len(pairs))
    return pd.DataFrame(rows), per_protein
