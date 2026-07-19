# ROGII Wellbore Geology Prediction

Reproducible experiments and Japanese research notes for the Kaggle ROGII Wellbore Geology Prediction competition.

## Colab quick start

1. Put the competition data in Google Drive:

   ```text
   MyDrive/kaggle/rogii/data/
   ├── train/
   ├── test/
   └── sample_submission.csv
   ```

2. Open [`notebooks/00_colab_setup.ipynb`](notebooks/00_colab_setup.ipynb) in Google Colab.
3. Set the repository URL to `https://github.com/Okada-N13/rogii.git`.
4. Run the notebook from the first cell. It installs the locked Python 3.12 environment, validates the data, creates fixed well folds, and runs an eight-well smoke test.
5. After the smoke test succeeds, remove `--limit-wells 8`, choose a new run ID, and run all 773 wells.

See [`docs/colab_workflow.md`](docs/colab_workflow.md) for the complete workflow.

After reproducing the anchor baseline, run [`notebooks/20_run_stage1_trends.ipynb`](notebooks/20_run_stage1_trends.ipynb). Measured Stage 1 results are recorded in [`docs/stage1_results.md`](docs/stage1_results.md).

For Stage 2, run [`notebooks/30_run_stage2_pf.ipynb`](notebooks/30_run_stage2_pf.ipynb) after the setup notebook. It evaluates the promoted particle-filter + guarded-trend blend. Measured results and leakage controls are recorded in [`docs/stage2_results.md`](docs/stage2_results.md).

For Stage 3, run the self-contained [`notebooks/40_run_stage3_residual.ipynb`](notebooks/40_run_stage3_residual.ipynb). It reuses Stage 2 when available and cross-fits the promoted residual sequence ensemble. Results are recorded in [`docs/stage3_results.md`](docs/stage3_results.md).

For Stage 4, run the self-contained [`notebooks/50_run_stage4_tail_path.ipynb`](notebooks/50_run_stage4_tail_path.ipynb). It adds a fixed uncertainty guard and a small deterministic trellis component. Results are recorded in [`docs/stage4_results.md`](docs/stage4_results.md).

Stage 5 is a self-contained spatial audit in [`notebooks/60_run_stage5_spatial_audit.ipynb`](notebooks/60_run_stage5_spatial_audit.ipynb). Its candidate is intentionally not promoted because it fails the leave-spatial-block and tail gates. See [`docs/stage5_results.md`](docs/stage5_results.md).

Generate the final honest Stage 4 CSV with the self-contained [`notebooks/70_generate_submission.ipynb`](notebooks/70_generate_submission.ipynb). Upload instructions and validation checks are documented in [`docs/submission_workflow.md`](docs/submission_workflow.md).

Stage 6 is the separate public-7.x positive control in [`notebooks/80_run_stage6_public7_reproduction.ipynb`](notebooks/80_run_stage6_public7_reproduction.ipynb). It downloads the attributed public Kaggle kernel at runtime, audits the generated CSV, and never mixes its public score with honest OOF. See [`docs/stage6_results.md`](docs/stage6_results.md).

## Local commands

```bash
uv sync --frozen
uv run pytest -q
uv run rogii-prepare --data-dir data --artifact-dir artifacts
uv run rogii-train \
  --config configs/experiment/baseline_anchor.yaml \
  --run-id baseline_anchor_v001 \
  --data-dir data \
  --artifact-dir artifacts
```

The competition dataset and generated experiment artifacts are intentionally excluded from Git.
Downloaded third-party public notebooks are also kept local and are not republished by this repository.
