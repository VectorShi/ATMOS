# MISATO evaluation

The evaluator reports four metrics for **ligand-only** and **pocket–ligand**
trajectories. Each target has 100 frames. All metrics use heavy atoms, including
protein sidechains and ligand atoms.

## Run

Use an environment containing the dependencies in
[requirements.txt](../../../requirements.txt), including NumPy, SciPy, pandas,
MDTraj, PyTorch, and Biotite. Evaluation runs on CPU. `PYTHON_BIN` can select the
Python executable; `--num-workers` controls the number of concurrent targets.

For a single inference run or a parallel-halves launch directory:

```bash
bash atmos/evaluation/misato/run_misato_evaluation.sh \
    --work-dir /path/to/inference-or-launch-directory \
    --output-dir /path/to/evaluation-output \
    --num-workers 6
```

A run contains `predictions/rank_<rank>_<pdb_id>.pkl.gz`. A parallel launch
contains two run directories with `predictions/` inside each. Targets from both
groups are evaluated together, and duplicate target IDs cause an error. Multiple
disjoint run directories can also follow `--work-dir`.

Prediction pickles supply both generated and reference coordinates, atom
identities, masks, and ligand bonds. The first 100 frames of each are converted
to PDB/XTC under the evaluation output directory. Each frame is rigidly aligned
to the preceding aligned frame; PDB coordinate rounding and XTC compression
apply before metric calculation. Input prediction files remain unchanged.

For an extracted trajectory package containing `trajectories/` and
`gt_trajectories/`, each with `<pdb_id>.pdb` and `<pdb_id>.xtc`:

```bash
bash atmos/evaluation/misato/run_misato_evaluation.sh \
    --work-dir /path/to/extracted-trajectory-package \
    --bond-reference-dir /path/to/misato/test_MD \
    --output-dir /path/to/evaluation-output
```

`--bond-reference-dir` contains the processed MISATO `<pdb_id>.pkl.gz`
bioassemblies. Their ligand bond tables are required because trajectory PDBs
may omit ligand connectivity. These files are included in `misato/test_MD.tar`
in the [ModelScope data package](../../../docs/assets.md#download);
extract the archive into `ATMOS_ROOT_DIR`.
The default bond reference directory is `$ATMOS_ROOT_DIR/misato/test_MD`
with repository `atmos_oss/` as the default asset root. No additional bond directory is needed
for prediction pickles. `--gt-dir` can override the reference PDB/XTC directory.
External prediction and reference XTC files must each contain exactly 100 frames.

The evaluator restricts inputs to the
[fixed MISATO test list](../../data/preprocessing/misato/misato_test_include_only_list.txt).
Use `--target-list` with a subset of these IDs for comparisons on a common target
set. Absent predictions are listed in the report; malformed inputs and metric
failures are recorded in `errors.json` and cause a nonzero exit without an
aggregate report. Prediction and reference heavy-atom identities and ordering
must match. Output directories must be new or empty.

## Atom selection and alignment

The final topology chain is the ligand; preceding chains contain the protein.
Hydrogens are removed before pocket selection and metric calculation. The
pocket is fixed from **reference frame 0**: include all atoms of every protein
residue whose nearest heavy-atom distance to any ligand heavy atom is **≤10 Å**.
Selection uses the atoms present in the supplied reference topology; missing
atoms or portions outside an inference crop are not reconstructed.

For RMSF, RMWD, and t-RMSD Error, each evaluated atom set is independently
superposed onto its own first frame, separately for prediction and reference.
Thus ligand-only alignment uses ligand atoms; pocket–ligand alignment uses
both pocket and ligand atoms with equal weights. Clash counts use interatomic
distances and require no alignment.

## Metrics

| Metric | Per-target calculation | Aggregation across targets |
| --- | --- | --- |
| Per-target RMSF r ↑ | Pearson correlation between predicted and reference per-atom RMSF vectors | Median |
| Root mean W2 distance ↓ | Root mean squared atom-wise Gaussian 2-Wasserstein distance | Median, Å |
| Steric clashes ↓ | Total qualifying atom-pair occurrences over all 100 generated frames | Mean, pairs per trajectory |
| t-RMSD Error ↓ | RMSE between predicted and reference RMSD-to-initial-frame time series | Mean, Å |

RMSF uses `mdtraj.rmsf(aligned_trajectory, aligned_trajectory[0])`: the reference
position is the first-frame position of each atom. It is not the atom's temporal
mean. Pearson correlation is computed over atoms within each target. Undefined
correlations remain NaN and are not silently dropped from the median.

For RMWD, fit a 3D Gaussian to each atom's aligned positional distribution using
population covariance (division by the number of frames). Combine the squared
mean-distance and covariance contributions, average over atoms, and take the
square root. A numerical exception or complex covariance contribution replaces
the covariance term for the entire target and scope by the reference RMS
fluctuation about its mean, `sqrt(mean_i trace(C_ref,i))`. This fallback is
recorded per target; it does not use the predicted covariance. RMWD is not
reported for `3m3c` and `6inz`; neither is in the fixed 100-target list.

A clash uses a distance **≤1.1 Å**. Ligand-only clashes count unordered ligand
heavy-atom pairs, excluding self-pairs and directly bonded pairs. Ligand bonds
are transferred from the bioassembly topology by residue name and atom name.
Repeated keys resolve to the last atom in topology order. The number of such
key collisions is recorded in `per_target.csv` for multi-residue ligands.
Pocket–ligand clashes count each pocket-protein/ligand heavy-atom pair once;
they do not include protein–protein or ligand–ligand pairs. Counts are summed
over frames, then averaged over targets, without division by the frame count.
Distances do not use periodic boundary wrapping.

For t-RMSD Error, compute each frame's RMSD to its own trajectory's first frame,
then take the RMSE between the two 100-point time series, including frame 0.

## Outputs

- `summary.txt`: two rows, one for each atom selection, and all four metrics.
- `summary.csv` and `summary.json`: metrics, units, aggregation rules, target
  coverage, and finite-value counts. JSON also lists RMWD fallback targets.
- `per_target.csv`: per-target metrics, atom counts, RMWD components and fallback
  reasons, and exclusion flags.
- `run_config.json`: exact input paths, evaluated/missing target IDs, thresholds,
  environment versions, and completion status.
- `errors.json`: target-level failures; empty on success.
- `trajectories/` and `gt_trajectories/`: converted inputs when evaluating
  prediction pickles.

Target and atom counts describe coverage; they are not evaluation metrics.
Undefined values appear as NaN in text/CSV and null in JSON. All four metrics
read the same prediction/reference pair for each target.
