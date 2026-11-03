# mdCATH evaluation

## Prepare reference data

The [ModelScope data package](../../../docs/assets.md#download)
includes the GT trajectories and crystal references in
`mdCATH_tempo/evaluation_references.tar.gz`. Set `ATMOS_ROOT_DIR` to the download
root, which defaults to repository `atmos_oss/`, and extract the archive there
to use these files directly. The preparation command below is only needed
when building references from raw data.

Prepare crystal references and GT trajectories from raw mdCATH H5 files and
preprocessed trajectory pickles. Run from the ATMOS repository root:

```bash
python -m atmos.evaluation.mdcath.prepare_inputs \
  --raw-h5-dir /path/to/mdCATH/data \
  --tempo-pkl-dir /path/to/mdCATH_tempo_test \
  --output-root /path/to/mdCATH_tempo \
  --max-workers 8
```

This writes `mdCATH_tempo_test_ref_pdb/<protein>/ref.pdb`,
`mdCATH_tempo_test_gt_trajectories/`, and a topology check CSV under the output
root. Crystal coordinates come from the H5 `pdbProteinAtoms` PDB, with hydrogens
and waters removed. Each GT trajectory contains the full preprocessed replica,
aligned frame-to-frame, with its first frame saved as the topology PDB. Omit the
preparation command's `--max-len` option to retain all GT frames.

To include predictions in the topology check, pass
`--pred-trajectories-dir /path/to/inference_run/trajectories`.

## Run

From the ATMOS repository root, use an environment containing the evaluation
dependencies and specify the GT trajectories and crystal references:

`--gt-dir` and `--ref-dir` default to the corresponding directories under
`$ATMOS_ROOT_DIR/mdCATH_tempo`, with repository `atmos_oss/` as the default
asset root. Explicit arguments override these paths.

```bash
PYTHON_BIN=python bash atmos/evaluation/mdcath/run_mdcath_evaluation.sh \
  --work-dir /path/to/inference_run \
  --gt-dir /path/to/mdCATH_tempo_test_gt_trajectories \
  --ref-dir /path/to/mdCATH_tempo_test_ref_pdb
```

The working directory can contain prediction files named
`predictions/rank_<rank>_mdcath_dataset_<protein>_0.pkl.gz`, or trajectories named
`trajectories/mdcath_dataset_<protein>_0.xtc` with matching PDB files. Each protein
must have one prediction trajectory. If both input directories are present,
predictions are converted by default. To use the existing trajectories:

```bash
PYTHON_BIN=python bash atmos/evaluation/mdcath/run_mdcath_evaluation.sh \
  --work-dir /path/to/inference_run \
  --gt-dir /path/to/mdCATH_tempo_test_gt_trajectories \
  --ref-dir /path/to/mdCATH_tempo_test_ref_pdb \
  --skip-trajectory-conversion \
  --output-dir /path/to/new_evaluation
```

From another directory, use the absolute path to the shell entry point.
`--output-dir` must be new or empty; its default is a timestamped subdirectory of
the working directory. Input files are preserved. Run with `--help` for all options.

GT must contain five replicas per protein, `mdcath_dataset_<protein>_{0..4}.xtc`,
and the topology `mdcath_dataset_<protein>_0.pdb`. The crystal reference directory
must contain `<protein>/ref.pdb`.

`1waae00` and `2noga01` are excluded because reference residue numbering and a
missing residue, respectively, prevent consistent evaluation. At least two
proteins are required for the correlation metrics.

Prediction conversion keeps the first `--max-len` frames (default: 400).
Predictions must provide at least this many frames; preconverted XTC files must
contain exactly this many. GT trajectories are kept at full length, with at least
`--max-len` frames in the first replica. Dynamics metrics assume the benchmark's
1 ns frame interval.

For inference with two parallel groups, collect their prediction files under
one `predictions/` directory before evaluating. The two groups must contain
disjoint proteins; pass the parent of the combined directory as `--work-dir`.

## Metrics

Ensemble metrics use Cα coordinates aligned to the crystal reference.
`MD pairwise RMSD` and `MD RMSF` describe GT, while `Pairwise RMSD` and `RMSF`
describe predictions.

| Metric | Aggregation |
| --- | --- |
| MD pairwise RMSD, Pairwise RMSD | Median over proteins, Å |
| Pairwise RMSD r | Pearson r across proteins |
| MD RMSF, RMSF | Median over pooled residues, Å |
| Global RMSF r | Pearson r over pooled residues |
| Per target RMSF r | Median of per-protein Pearson r |
| RMWD, RMWD trans, RMWD var | Median over proteins, Å |
| MD PCA W2, Joint PCA W2 | Median over proteins, Å |
| PC sim > 0.5 % | Percentage of proteins |
| Weak contacts J, Transient contacts J | Median over proteins |
| Weak contacts nans, Transient contacts nans | Fraction of undefined values |
| Exposed residue J, Exposed MI matrix rho | Median over proteins |
| rmse_of_rmsd | Mean per-protein Cα RMSD-series RMSE against GT, Å |
| val_ca | Mean and sample standard deviation across proteins |
| msm_tm_rmse | Mean and sample standard deviation across proteins |
| TIC0 decorrelation Pearson r | Pearson r between predicted and GT decay times |

The two exposed-sidechain metrics are undefined in CA-only analysis. They remain
NaN in text/CSV and null in JSON, with zero contributing proteins.

`rmse_of_rmsd` compares the first `max_len` frames of the prediction and first GT
replica. Each Cα trajectory is aligned to its own first frame to obtain its RMSD
time series. `val_ca` is the fraction of frames with no Cα clashes below 3.0 Å
and no within-chain adjacent Cα distances above 4.19 Å.

TICA uses sine/cosine backbone torsions, five GT replicas, lag 1, dimension 2, and
kinetic-map scaling. The decay time is the first lag at which autocorrelation is
strictly below 0.5, searching up to 100 lags. The GT trace uses the first
`max_len` frames of the first replica.
The prediction fallback is `max_len` if no crossing is found. Missing GT
crossings are excluded from Pearson r and reflected in its `n` column.

MSM uses 100 microstates, K-means seed 137, 10 metastable states, and lag 1.
GT replicas are concatenated for transition counting, including transitions
across replica boundaries. `msm_tm_rmse` is the unweighted RMSE between the two
row-normalized 10-by-10 transition matrices; states with no outgoing transitions
have zero rows.

## Outputs

- `summary.txt`, `summary.csv`, `summary.json`: the complete metric set,
  aggregation, units, contributing protein counts, and applicable standard deviations.
- `per_protein.csv`: protein IDs, ensemble quantities, temporal RMSE, Cα
  validity, MSM transition RMSE, and the two decay times used for Pearson r.
- `run_config.json`: input paths, protein IDs, parameters, and environment versions.
- `errors.json`: per-protein dynamics failures; empty on a successful run.
- `ensemble.log` and optional `conversion.log`: ensemble and conversion logs.
- `out_ca_only.pkl`: detailed ensemble quantities.

The number of evaluated proteins is report metadata: `protein_count` in
`summary.json` and a header in `summary.txt`. Each metric's `n` column records
the number of contributing proteins; for pooled-residue metrics it records the
number of input proteins. `std` is the sample standard deviation (`ddof=1`) for
`val_ca` and `msm_tm_rmse`.

Evaluation failures cause a nonzero exit status. Summary files are written only
after all proteins have completed the ensemble and dynamics calculations.
