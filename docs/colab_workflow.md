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
