"""Backbone torsion features and discretization into MSM metastable states."""

import numpy as np
import deeptime as dt
import mdtraj as md

SELECTION = "symbol == C or symbol == N or symbol == S"


def wrap(array):
    return np.sin(array), np.cos(array)


def reduced_tica_features(trajectory, use_distances=False, selection=SELECTION, contact_pairs=None):
    _, phi = md.compute_phi(trajectory)
    _, psi = md.compute_psi(trajectory)
    torus = np.concatenate([phi, psi], axis=1)
    dihedrals = np.concatenate([*wrap(torus)], axis=1)
    return dihedrals, None


def get_kmeans(feats, n_clusters=100, max_iter=100):
    kmeans = dt.clustering.KMeans(n_clusters=n_clusters, max_iter=max_iter, fixed_seed=137).fit_fetch(feats)
    return kmeans, kmeans.transform(feats)


class KMeansWrapper:
    def __init__(self, kmeans, active_states):
        self.kmeans = kmeans
        self.active_states = active_states
        self.microstate_map = self.assign_to_active_state()

    def assign_to_active_state(self) -> np.ndarray:
        cluster_centers = self.kmeans.cluster_centers   # (n_micro, dim)
        active_centers = cluster_centers[self.active_states]    # (n_active, dim)

        diff = cluster_centers[:, None, :] - active_centers[None, :, :]  # (n_micro, n_active, dim)
        dist_sq = np.sum(diff**2, axis=2) # (n_micro, n_active)
        nearest_rel_idx = np.argmin(dist_sq, axis=1)  # (n_micro,)

        return nearest_rel_idx

    def transform(self, feats):
        trans_feats = self.kmeans.transform(feats)
        return self.microstate_map[trans_feats]


def get_msm(feats, lagtime=1000, nstates=10, n_clusters=100):
    msm = dt.markov.msm.MaximumLikelihoodMSM(lagtime=lagtime).fit_fetch(feats)
    pcca = msm.pcca(nstates)
    print(f"pcca assignments: {len(pcca.assignments)}")
    return msm, pcca


def msm_discretize(feats, kmeans, pcca):
    return pcca.assignments[kmeans.transform(feats)]
