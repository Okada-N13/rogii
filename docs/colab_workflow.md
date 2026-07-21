# Colab experiment workflow

## Principles

- Git stores code, configuration, notebooks, and `uv.lock`.
- Google Drive stores competition data and persistent experiment artifacts.
- `/content` stores the repository, copied input data, and temporary caches while a runtime is alive.
- Model and feature logic belongs in `src/rogii`, not in notebook cells.
- Every experiment is launched from a YAML config and writes to a new run directory.

## Google Drive layout

```text
MyDrive/kaggle/rogii/
├── data/
│   ├── train/
│   ├── test/
│   └── sample_submission.csv
└── artifacts/
```

The directory referenced by `ROGII_DATA_DIR` must directly contain `train/` and `test/`.

## First Colab run

Open `notebooks/00_colab_setup.ipynb`, set `REPOSITORY_URL`, and run all cells. It will:

1. mount Google Drive;
2. clone a specific Git revision into `/content/ROGII`;
3. install uv;
4. create a Python 3.12 environment that can see Colab's CUDA-enabled packages;
5. sync the locked project dependencies;
6. copy the CSV data to local runtime storage;
7. validate the data and create fixed five-fold well assignments.

Then open `notebooks/10_run_anchor_baseline.ipynb` in the same runtime and run the baseline. The primary flat baseline holds the last visible TVT constant. A separate `baseline_surface_anchor.yaml` is retained as a diagnostic because holding `TVT+Z` constant performs much worse on the supplied training masks.

## CLI equivalents

```bash
uv run rogii-prepare \
  --data-dir /content/rogii-data \
  --artifact-dir /content/drive/MyDrive/kaggle/rogii/artifacts

uv run rogii-train \
  --config configs/experiment/baseline_anchor.yaml \
  --run-id baseline_anchor_v001 \
  --data-dir /content/rogii-data \
  --artifact-dir /content/drive/MyDrive/kaggle/rogii/artifacts
```

Use a new `run-id` for every run. The training command refuses to overwrite a non-empty run directory.

## Smoke test

Before processing all 773 wells, use a small subset:

```bash
uv run rogii-prepare --limit-wells 8 --skip-fingerprint
uv run rogii-train --run-id smoke_anchor --limit-wells 8
```

After the smoke test succeeds, remove `--limit-wells`. The complete baseline writes:

```text
artifacts/<run-id>/
├── config.yaml
├── environment.json
├── folds.parquet
├── metrics.json
├── oof.parquet
├── per_well_metrics.parquet
├── run.log
└── well_stats.parquet
```

## Stage 1 trend blend

Once `baseline_anchor_full_v001` exists, open `notebooks/20_run_stage1_trends.ipynb`. It pulls the latest code, runs the promoted fixed trend blend on all 773 wells, and compares it with the anchor baseline. The expected development OOF RMSE is approximately `15.700508`.

The standalone linear and quadratic configs are diagnostic controls and are not promoted models. See `docs/stage1_results.md` for their measured failure modes and the nested validation result.

## Stage 2 particle filter

Open `notebooks/30_run_stage2_pf.ipynb`. It is self-contained and can initialize a fresh Colab runtime by mounting Drive, cloning or updating the repository, syncing dependencies, and copying the input data. It then runs an eight-well smoke test before the complete 773-well evaluation. It is not necessary to rerun the Stage 1 notebook first: the promoted Stage 1 component is included in `configs/experiment/pf_trend_blend.yaml`. An existing Stage 1 artifact is used only for the final comparison report.

The promoted model is a fixed blend of 75% two-batch multi-seed particle filter and 25% Stage 1 guarded trend blend. The PF averages two independent 16-seed batches to reduce its material random-seed variance. It uses the complete visible GR/trajectory sequence, the known `TVT_input` prefix, and the paired typewell, and never reads hidden target TVT while predicting. The expected development OOF RMSE is approximately `12.565438`; each 16-seed full CPU pass took about 15 minutes on the local reference machine, so allow roughly 30--45 minutes in Colab.

## Stage 3 residual sequence ensemble

Open `notebooks/40_run_stage3_residual.ipynb`. Like every notebook from Stage 2 onward, it is self-contained: it mounts Drive, clones or updates the repository, syncs dependencies, copies data, and creates its prerequisite Stage 2 run automatically when missing. When `stage2_pf_trend_blend_full_v001` already exists, it is reused without rerunning PF.

Stage 3 cross-fits a two-seed lightweight residual ensemble on multi-scale GR, PF surface, typewell mismatch, trajectory, and known-prefix robust-trend features. No hidden TVT is used to construct features. The native Colab reproduction OOF RMSE is `12.339250`, versus `12.565438` for Stage 2; the locally composed development reference was `12.299725`. With Stage 2 already present, allow approximately 3--10 minutes on a Colab CPU runtime.

## Stage 4 tail guard and trellis

Open `notebooks/50_run_stage4_tail_path.ipynb`. It is self-contained and creates Stage 2 and Stage 3 automatically only when their expected Drive artifacts are missing. With `stage3_residual_hgb_full_v001` already present, only the deterministic Stage 4 pass runs.

Stage 4 caps the Stage 3 correction to ±1 ft for the 20% of wells with the highest mean PF seed uncertainty. It then adds 10% of a deterministic GR/typewell trellis correction. The local reference OOF improved from `12.299725` to `12.204505`, while median, P90, and maximum well RMSE also improved. Because the locally composed Stage 3 diagnostics produced a slightly different score from native Colab, treat the first native Colab Stage 4 result as the canonical reproduction value. Allow approximately 2--8 minutes on a Colab CPU runtime when Stage 3 is present.

The native Colab Stage 4 reproduction scored `12.228593`, improving its Stage 3 input by `0.110657`. Its well-paired bootstrap interval was `[-0.157746, -0.037241]`.

## Stage 5 spatial audit

Open `notebooks/60_run_stage5_spatial_audit.ipynb`. It is self-contained and reuses Stage 4 when present. The notebook evaluates the same nearest-well residual correction under ordinary well folds, six geographic holdout blocks, and a shuffled-target control, then prints every promotion gate.

The local candidate improved ordinary OOF by `0.072531`, but improved spatial-block OOF by only `0.009404`, worsened two of six geographic blocks, and increased the worst-10% SSE share. It is therefore an audit result, not a promoted replacement for Stage 4. Allow approximately 2--8 minutes when Stage 4 is already present.

## Generate a submission

Open `notebooks/70_generate_submission.ipynb`. It is self-contained, recreates only missing prerequisites, applies the promoted Stage 4 stack to the test wells, and saves validated primary and secondary CSV files under `MyDrive/kaggle/rogii/submissions/stage4_submission_v001/`.

Upload `submission.csv` as the primary Stage 4 entry. `submission_stage3.csv` is a secondary control. See `docs/submission_workflow.md` for the exact checks and upload procedure.

## Stage 6B stable PF64 bimodal overlay

Open `notebooks/90_run_stage6_pf128_heel_mha.ipynb` directly; `00_colab_setup.ipynb` does not need to be run first. The Notebook mounts Drive, clones or updates the repository, syncs dependencies, copies the data, runs an eight-well smoke check, and then performs the full experiment.

The expensive diagnostic uses four independent 16-seed × 256-particle batches. Four wells run concurrently through `runtime.n_jobs: 4`; this produced byte-identical smoke metrics to sequential execution locally. The diagnostic supplies bimodal shifts only, while the promoted Stage 4 prediction remains the base. Completed artifacts in Drive are reused on rerun. If Colab stops during a run, change the affected `*_RUN_ID` suffix before restarting because experiment commands deliberately refuse to overwrite partial artifacts.

The final cell compares the MHA overlay with `stage4_tail_path_full_v001`. The local reference improved by `0.032720`, although P90 worsened slightly and the paired-bootstrap interval crossed zero. Return the native Colab dictionary before proceeding to the Kaggle execution Notebook; the local number is not a leaderboard estimate.

## Stage 7 public pretrained residual gate

Open `notebooks/100_colab_public_residual_gate.ipynb` directly. It is standalone and does not require `00_colab_setup.ipynb`. Colab is the correct environment for this stage because it needs Internet access to fetch the ravaghi pretrained artifact and enough time to reconstruct full OOF diagnostics; it does not produce a submission.

The Notebook rebuilds the ravaghi public-stack OOF from five saved `koolbox.Trainer` objects, cross-fits a conservative residual correction, and evaluates ordinary well folds, geographic-block refits, paired-well bootstrap, P90, and worst-well concentration. It saves the validation report and trained correction models to Drive.

Only continue to a Kaggle Internet-OFF inference Notebook when the final result says `promoted: true`. A small `LIMIT_WELLS` run is only a wiring check and cannot authorize a submission. See `docs/stage7_public_residual_gate.md` for the exact gates and limitations.

## Stage 7B nested physics gate

Stage 7's learned HGB correction worsened all five folds and was rejected. Open `notebooks/110_colab_public_physics_gate.ipynb` for the next experiment. It reuses only Stage 7's unmodified public base OOF and spatial fold map, so it does not download or deserialize the public pretrained artifact again.

For every outer fold, Stage 7B selects a small visible-prefix `TVT + Z` polynomial correction using only the other wells. It repeats the nested selection across geographic blocks and applies the same bootstrap and tail gates. Continue to Kaggle integration only if the final dictionary reports `promoted: true`.

## Stage 7C public package OOF audit

Stage 7B also failed its nested gates. Open `notebooks/120_colab_public_package_audit.ipynb` to inspect the fleongg and pilkwang public packages before attempting a model-level blend. The Notebook downloads both public Datasets, inventories OOF/fold/manifest assets, and verifies row IDs against the Stage 7 ravaghi base OOF.

This is an inexpensive CPU metadata run, not an experiment or submission. Return its final summary before building the nested blend. A same-length prediction array is not treated as OOF unless its ordering and fold provenance can be demonstrated.

## Stage 7D verified public OOF blend

The package audit found no OOF in the fleongg inference package, so that branch is excluded from honest blend fitting. The pilkwang package includes complete family, TCN, raw-blend, and postprocessed OOF plus an ID/ground-truth table.

Open `notebooks/130_colab_public_verified_blend_gate.ipynb`. It verifies exact row IDs and target values, then nested-selects a pilkwang branch and a 1--40% weight against the ravaghi base. It repeats selection across geographic blocks and applies the standard bootstrap/tail gates. Do not port the reported all-OOF inference spec unless `promoted` is true.

## Stage 7E spatially robust public blend

Stage 7D strongly improved ordinary OOF and aggregate spatial RMSE but failed the required spatial-block consistency. Open `notebooks/140_colab_public_robust_blend_gate.ipynb` for a restricted confirmation: only the postprocessed pilkwang branch and 20/30/40% weights remain, and a weight is eligible only if it does not hurt any inner fold or block.

This uses the same OOF population and is therefore a robustness analysis rather than a fresh independent test. Even if it passes, Kaggle integration replaces only the ravaghi branch; it does not apply the reported package weight directly to the complete Safe MHA submission.

## Stage 11 independent multi-cut delta-U baseline

Open `notebooks/250_run_stage11_multicut_delta_u.ipynb` directly. It is standalone and begins the independent 5-point research track; it does not require any public artifact or earlier experiment run.

The notebook creates four pseudo-test cuts per training well, predicts low-frequency `U = TVT + Z` slope and curvature corrections, and audits the same model under ordinary well folds, geographic blocks, and typewell-signature blocks. It also runs a hidden-suffix target invariance check before fitting. Stage 11 uses CPU only; choose a high-RAM Colab session when available and leave `LIMIT_WELLS = None` for the decision run.

No submission is generated. Return the final validation dictionary and fold-delta table. A true promotion result authorizes the Stage 12 GR-alignment benchmark, not a Kaggle submission.
