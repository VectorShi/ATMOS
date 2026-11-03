"""Compare generated mdCATH ensembles with MD and crystal references."""

import argparse
import os
from pathlib import Path

DEFAULT_DATA = Path(os.environ.get(
    "ATMOS_ROOT_DIR", str(Path(__file__).resolve().parents[3] / "atmos_oss")
)) / "mdCATH_tempo"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--result_dir', type=str, required=True)
parser.add_argument(
    '--gt_dir',
    type=str,
    default=str(DEFAULT_DATA / 'mdCATH_tempo_test_gt_trajectories'),
    help='GT trajectories under ATMOS_ROOT_DIR/mdCATH_tempo (repository atmos_oss/ if unset).',
)
parser.add_argument(
    '--ref_dir',
    type=str,
    default=str(DEFAULT_DATA / 'mdCATH_tempo_test_ref_pdb'),
    help='Crystal references under ATMOS_ROOT_DIR/mdCATH_tempo (repository atmos_oss/ if unset).',
)
parser.add_argument('--bb_only', action='store_true')
parser.add_argument('--ca_only', action='store_true')
parser.add_argument('--num_workers', type=int, default=1)

args = parser.parse_args()
from sklearn.decomposition import PCA
import mdtraj, pickle, tqdm
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linear_sum_assignment


def get_pca(xyz):
    traj_reshaped = xyz.reshape(xyz.shape[0], -1)
    pca = PCA(n_components=min(traj_reshaped.shape))
    coords = pca.fit_transform(traj_reshaped)
    return pca, coords


def get_rmsds(traj1, traj2, broadcast=False):
    n_atoms = traj1.shape[1]
    traj1 = traj1.reshape(traj1.shape[0], n_atoms * 3)
    traj2 = traj2.reshape(traj2.shape[0], n_atoms * 3)
    if broadcast:
        traj1, traj2 = traj1[:,None], traj2[None]
    distmat = np.square(traj1 - traj2).sum(-1)**0.5 / n_atoms**0.5 * 10
    return distmat


def condense_sidechain_sasas(sasas, top):
    assert top.n_residues > 1

    if top.n_atoms != sasas.shape[1]:
        raise ValueError(
            f"The number of atoms in top ({top.n_atoms}) didn't match the "
            f"number of SASAs provided ({sasas.shape[1]}). Make sure you "
            f"computed atom-level SASAs (mode='atom') and that you've passed "
            "the correct topology file and array of SASAs"
        )

    sc_mask = np.array([a.name not in ['CA', 'C', 'N', 'O', 'OXT'] for a in top.atoms])
    res_id = np.array([a.residue.index for a in top.atoms])

    rsd_sasas = np.zeros((sasas.shape[0], top.n_residues), dtype='float32')

    for i in range(top.n_residues):
        rsd_sasas[:, i] = sasas[:, sc_mask & (res_id == i)].sum(1)
    return rsd_sasas


def sasa_mi(sasa):
    N, L = sasa.shape
    joint_probs = np.zeros((L, L, 2, 2))

    joint_probs[:,:,1,1] = (sasa[:,:,None] & sasa[:,None,:]).mean(0)
    joint_probs[:,:,1,0] = (sasa[:,:,None] & ~sasa[:,None,:]).mean(0)
    joint_probs[:,:,0,1] = (~sasa[:,:,None] & sasa[:,None,:]).mean(0)
    joint_probs[:,:,0,0] = (~sasa[:,:,None] & ~sasa[:,None,:]).mean(0)

    marginal_probs = np.stack([1-sasa.mean(0), sasa.mean(0)], -1)
    indep_probs = marginal_probs[None,:,None,:] * marginal_probs[:,None,:,None]
    mi = np.nansum(joint_probs * np.log(joint_probs / indep_probs), (-1, -2))
    mi[np.arange(L), np.arange(L)] = 0
    return mi


def get_mean_covar(xyz):
    mean = xyz.mean(0)
    xyz = xyz - mean
    covar = (xyz[...,None] * xyz[...,None,:]).mean(0)
    return mean, covar


def sqrtm(M):
    D, P = np.linalg.eig(M)
    out = (P * np.sqrt(D[:,None])) @ np.linalg.inv(P)
    return out


def get_wasserstein(distmat, p=2):
    assert distmat.shape[0] == distmat.shape[1]
    distmat = distmat ** p
    row_ind, col_ind = linear_sum_assignment(distmat)
    return distmat[row_ind, col_ind].mean() ** (1/p)


def align_tops(top1, top2):
    names1 = [repr(a) for a in top1.atoms]
    names2 = [repr(a) for a in top2.atoms]

    intersection = [nam for nam in names1 if nam in names2]

    mask1 = [names1.index(nam) for nam in intersection]
    mask2 = [names2.index(nam) for nam in intersection]
    return mask1, mask2

def align_tops_three(top1, top2, top3):
    names1 = [repr(a) for a in top1.atoms]
    names2 = [repr(a) for a in top2.atoms]
    names3 = [repr(a) for a in top3.atoms]

    intersection = [nam for nam in names1 if nam in names2 and nam in names3]
    mask1 = [names1.index(nam) for nam in intersection]
    mask2 = [names2.index(nam) for nam in intersection]
    mask3 = [names3.index(nam) for nam in intersection]
    return mask1, mask2, mask3


def filter_zero_coordinate_atoms(traj, label, atol=1e-6):
    zero_atom_mask = np.all(np.abs(traj.xyz) <= atol, axis=(0, 2))
    if not zero_atom_mask.any():
        return

    removed_atoms = [repr(atom) for atom in traj.top.atoms if zero_atom_mask[atom.index]]
    keep_atom_indices = np.flatnonzero(~zero_atom_mask)
    if keep_atom_indices.size == 0:
        raise ValueError(f"All atoms in {label} have zero coordinates.")

    preview = ", ".join(removed_atoms[:10])
    suffix = "..." if len(removed_atoms) > 10 else ""
    print(f"Removing {len(removed_atoms)} zero-coordinate atoms from {label}: {preview}{suffix}")
    traj.atom_slice(keep_atom_indices, True)


def main(one_domain):
    name = one_domain['dname']
    print('Analyzing', name)
    out = {}
    ref_aa = mdtraj.load(one_domain['ref_pdb'])

    print('Loading reference trajectory')
    traj_aa = None
    for fpath in one_domain['gt']:
        topfile = one_domain['gt_top']
        if traj_aa is None:
            traj_aa = mdtraj.load(fpath, top=topfile)
        else:
            traj_aa = traj_aa + mdtraj.load(fpath, top=topfile)

    print(f'Loaded {traj_aa.n_frames} reference frames')

    print('Loading Pred conformers')
    aftraj_aa = None
    for fpath in one_domain['pred']:
        topfile = one_domain['pred_top']
        if aftraj_aa is None:
            aftraj_aa = mdtraj.load(fpath, top=topfile)
        else:
            aftraj_aa = aftraj_aa + mdtraj.load(fpath, top=topfile)

    print(f'Loaded {aftraj_aa.n_frames} predicted conformers')
    print(f'Reference has {traj_aa.n_atoms} atoms')
    print(f'Crystal has {ref_aa.n_atoms} atoms')
    print(f'Prediction has {aftraj_aa.n_atoms} atoms')

    print('Removing hydrogens')

    traj_aa.atom_slice([a.index for a in traj_aa.top.atoms if a.element.symbol != 'H'], True)
    ref_aa.atom_slice([a.index for a in ref_aa.top.atoms if a.element.symbol != 'H'], True)
    aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.element.symbol != 'H'], True)

    print(f'Reference has {traj_aa.n_atoms} atoms')
    print(f'Crystal has {ref_aa.n_atoms} atoms')
    print(f'Prediction has {aftraj_aa.n_atoms} atoms')

    if args.bb_only:
        traj_aa.atom_slice([a.index for a in traj_aa.top.atoms if a.name in ['CA', 'C', 'N', 'O', 'OXT']], True)
        ref_aa.atom_slice([a.index for a in ref_aa.top.atoms if a.name in ['CA', 'C', 'N', 'O', 'OXT']], True)
        aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.name in ['CA', 'C', 'N', 'O', 'OXT']], True)

    elif args.ca_only:
        traj_aa.atom_slice([a.index for a in traj_aa.top.atoms if a.name == 'CA'], True)
        ref_aa.atom_slice([a.index for a in ref_aa.top.atoms if a.name == 'CA'], True)
        aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.name == 'CA'], True)

    filter_zero_coordinate_atoms(traj_aa, 'reference trajectory')
    filter_zero_coordinate_atoms(ref_aa, 'crystal reference')
    filter_zero_coordinate_atoms(aftraj_aa, 'predicted conformers')

    ref_traj_mask, ref_aa_mask, afmask = align_tops_three(traj_aa.top, ref_aa.top, aftraj_aa.top)

    traj_aa.atom_slice(ref_traj_mask, True)
    ref_aa.atom_slice(ref_aa_mask, True)
    aftraj_aa.atom_slice(afmask, True)

    print(f'Aligned on {aftraj_aa.n_atoms} atoms')

    np.random.seed(137)
    RAND1 = np.random.randint(0, traj_aa.n_frames, aftraj_aa.n_frames)
    RAND2 = np.random.randint(0, traj_aa.n_frames, aftraj_aa.n_frames)
    RAND1K = np.random.randint(0, traj_aa.n_frames, 1000)

    traj_aa.superpose(ref_aa)
    aftraj_aa.superpose(ref_aa)

    out['ca_mask'] = ca_mask = [a.index for a in traj_aa.top.atoms if a.name == 'CA']
    traj = traj_aa.atom_slice(ca_mask, False)
    ref = ref_aa.atom_slice(ca_mask, False)
    aftraj = aftraj_aa.atom_slice(ca_mask, False)
    print(f'Sliced {aftraj.n_atoms} C-alphas')

    traj.superpose(ref)
    aftraj.superpose(ref)


    n_atoms = aftraj.n_atoms

    print(f'Doing PCA')

    ref_pca, ref_coords = get_pca(traj.xyz)
    af_coords_ref_pca = ref_pca.transform(aftraj.xyz.reshape(aftraj.n_frames, -1))
    seed_coords_ref_pca = ref_pca.transform(ref.xyz.reshape(1, -1))

    af_pca, af_coords = get_pca(aftraj.xyz)
    ref_coords_af_pca = af_pca.transform(traj.xyz.reshape(traj.n_frames, -1))
    seed_coords_af_pca = af_pca.transform(ref.xyz.reshape(1, -1))

    joint_pca, _ = get_pca(np.concatenate([traj[RAND1].xyz, aftraj.xyz]))
    af_coords_joint_pca = joint_pca.transform(aftraj.xyz.reshape(aftraj.n_frames, -1))
    ref_coords_joint_pca = joint_pca.transform(traj.xyz.reshape(traj.n_frames, -1))
    seed_coords_joint_pca = joint_pca.transform(ref.xyz.reshape(1, -1))

    out['ref_variance'] = ref_pca.explained_variance_ / n_atoms * 100
    out['af_variance'] = af_pca.explained_variance_ / n_atoms * 100
    out['joint_variance'] = joint_pca.explained_variance_ / n_atoms * 100

    out['af_rmsf'] = mdtraj.rmsf(aftraj_aa, ref_aa) * 10
    out['ref_rmsf'] = mdtraj.rmsf(traj_aa, ref_aa) * 10

    print(f'Computing atomic EMD')
    ref_mean, ref_covar = get_mean_covar(traj_aa[RAND1K].xyz)
    af_mean, af_covar = get_mean_covar(aftraj_aa.xyz)
    out['emd_mean'] = (np.square(ref_mean - af_mean).sum(-1) ** 0.5) * 10
    try:
        out['emd_var'] = (np.trace(ref_covar + af_covar - 2*sqrtm(ref_covar @ af_covar), axis1=1,axis2=2) ** 0.5) * 10
    except:
        out['emd_var'] = np.trace(ref_covar) ** 0.5 * 10


    print(f'Analyzing SASA')
    sasa_thresh = 0.02
    af_sasa = mdtraj.shrake_rupley(aftraj_aa, probe_radius=0.28)
    af_sasa = condense_sidechain_sasas(af_sasa, aftraj_aa.top)
    ref_sasa = mdtraj.shrake_rupley(traj_aa[RAND1K], probe_radius=0.28)
    ref_sasa = condense_sidechain_sasas(ref_sasa, traj_aa.top)
    crystal_sasa = mdtraj.shrake_rupley(ref_aa, probe_radius=0.28)
    out['crystal_sasa'] = condense_sidechain_sasas(crystal_sasa, ref_aa.top)

    out['ref_sa_prob'] = (ref_sasa > sasa_thresh).mean(0)
    out['af_sa_prob'] = (af_sasa > sasa_thresh).mean(0)
    out['ref_mi_mat'] = sasa_mi(ref_sasa > sasa_thresh)
    out['af_mi_mat'] = sasa_mi(af_sasa > sasa_thresh)

    ref_distmat = np.linalg.norm(traj[RAND1].xyz[:,None,:] - traj[RAND1].xyz[:,:,None], axis=-1)
    af_distmat = np.linalg.norm(aftraj.xyz[:,None,:] - aftraj.xyz[:,:,None], axis=-1)

    out['ref_contact_prob'] = (ref_distmat < 0.8).mean(0)
    out['af_contact_prob'] = (af_distmat < 0.8).mean(0)
    out['crystal_distmat'] = np.linalg.norm(ref.xyz[0,None,:] - ref.xyz[0,:,None], axis=-1)

    out['ref_mean_pairwise_rmsd'] = get_rmsds(traj[RAND1].xyz, traj[RAND2].xyz, broadcast=True).mean()
    out['af_mean_pairwise_rmsd'] = get_rmsds(aftraj.xyz, aftraj.xyz, broadcast=True).mean()

    out['ref_rms_pairwise_rmsd'] = np.square(get_rmsds(traj[RAND1].xyz, traj[RAND2].xyz, broadcast=True)).mean() ** 0.5
    out['af_rms_pairwise_rmsd'] = np.square(get_rmsds(aftraj.xyz, aftraj.xyz, broadcast=True)).mean() ** 0.5

    out['ref_self_mean_pairwise_rmsd'] = get_rmsds(traj[RAND1].xyz, traj[RAND1].xyz, broadcast=True).mean()
    out['ref_self_rms_pairwise_rmsd'] = np.square(get_rmsds(traj[RAND1].xyz, traj[RAND1].xyz, broadcast=True)).mean() ** 0.5

    out['cosine_sim'] = (ref_pca.components_[0] * af_pca.components_[0]).sum()


    def get_emd(ref_coords1, ref_coords2, af_coords, seed_coords, K=None):
        if len(ref_coords1.shape) == 3:
            ref_coords1 = ref_coords1.reshape(ref_coords1.shape[0], -1)
            ref_coords2 = ref_coords2.reshape(ref_coords2.shape[0], -1)
            af_coords = af_coords.reshape(af_coords.shape[0], -1)
            seed_coords = seed_coords.reshape(seed_coords.shape[0], -1)
        if K is not None:
            ref_coords1 = ref_coords1[:,:K]
            ref_coords2 = ref_coords2[:,:K]
            af_coords = af_coords[:,:K]
            seed_coords = seed_coords[:,:K]
        emd = {}
        emd['ref|ref mean'] = (np.square(ref_coords1 - ref_coords1.mean(0)).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10

        distmat = np.square(ref_coords1[:,None] - ref_coords2[None]).sum(-1)
        distmat = distmat ** 0.5 / n_atoms ** 0.5 * 10
        emd['ref|ref2'] = get_wasserstein(distmat)
        emd['ref mean|ref2 mean'] = np.square(ref_coords1.mean(0) - ref_coords2.mean(0)).sum() ** 0.5 / n_atoms ** 0.5 * 10

        distmat = np.square(ref_coords1[:,None] - af_coords[None]).sum(-1)
        distmat = distmat ** 0.5 / n_atoms ** 0.5 * 10
        emd['ref|af'] = get_wasserstein(distmat)
        emd['ref mean|af mean'] = np.square(ref_coords1.mean(0) - af_coords.mean(0)).sum() ** 0.5 / n_atoms ** 0.5 * 10

        emd['ref|seed'] = (np.square(ref_coords1 - seed_coords).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10
        emd['ref mean|seed'] = (np.square(ref_coords1.mean(0) - seed_coords).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10

        emd['af|seed'] = (np.square(af_coords - seed_coords).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10
        emd['af|af mean'] = (np.square(af_coords - af_coords.mean(0)).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10
        emd['af mean|seed'] = (np.square(af_coords.mean(0) - seed_coords).sum(-1)).mean()**0.5 / n_atoms ** 0.5 * 10
        return emd

    K=2
    out[f'EMD,ref'] = get_emd(ref_coords[RAND1], ref_coords[RAND2], af_coords_ref_pca, seed_coords_ref_pca, K=K)
    out[f'EMD,af2'] = get_emd(ref_coords_af_pca[RAND1], ref_coords_af_pca[RAND2], af_coords, seed_coords_af_pca, K=K)
    out[f'EMD,joint'] = get_emd(ref_coords_joint_pca[RAND1], ref_coords_joint_pca[RAND2], af_coords_joint_pca, seed_coords_joint_pca, K=K)
    return name, out


def get_all_ids(result_dir, gt_data_dir, ref_dir):
    all_fnames = os.listdir(result_dir)

    rm_ext = lambda x: os.path.splitext(x)[0]
    all_fnames = [rm_ext(fname) for fname in all_fnames]
    all_fnames = list(set(all_fnames))

    get_domain = lambda x: x.split('_')[-2]
    all_ids = {}
    for fname in all_fnames:
        dname = get_domain(fname)
        if dname not in all_ids.keys():
            all_ids[dname] = {"dname": dname, "pred": []}

        all_ids[dname]['pred'].append(
            os.path.join(result_dir, f"{fname}.xtc")
        )

    for dname in all_ids.keys():
        all_ids[dname]['pred'] = sorted(all_ids[dname]['pred'])
        all_ids[dname]['pred_top'] = all_ids[dname]['pred'][0][:-3] + "pdb"

        all_ids[dname]['gt'] = [
            os.path.join(gt_data_dir, f"mdcath_dataset_{dname}_{i}.xtc") for i in range(5)
        ]
        all_ids[dname]['gt_top'] = os.path.join(gt_data_dir, f"mdcath_dataset_{dname}_0.pdb")

    for dname in all_ids.keys():
        all_ids[dname]['ref_pdb'] = os.path.join(
            ref_dir, dname, "ref.pdb"
        )

    all_ids = list(all_ids.values())
    return all_ids


def error_packed_main(one_domain):
    try:
        return main(one_domain)
    except:
        name = one_domain['dname']
        return name, {}

if __name__ == "__main__":
    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map

    all_ids = get_all_ids(args.result_dir, args.gt_dir, args.ref_dir)

    out = dict(tqdm.tqdm(__map__(error_packed_main, all_ids), total=len(all_ids)))
    if args.num_workers > 1:
        p.__exit__(None, None, None)

    del_k = []
    for k in out.keys():
        if len(out[k]) == 0:
            print(k)
            del_k.append(k)

    for k in del_k: del out[k]

    insert_code = ""
    if args.bb_only:
        insert_code = "_bb_only"
    elif args.ca_only:
        insert_code = "_ca_only"
    save_dir = os.path.dirname(args.result_dir)
    with open(f"{save_dir}/out{insert_code}.pkl", 'wb') as f:
        f.write(pickle.dumps(out))
