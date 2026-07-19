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
