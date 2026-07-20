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

Stage 6 uses an attributed public Kaggle kernel as a separate public-7.x positive control. Formal submission must be made by copying and committing that Notebook on Kaggle; [`notebooks/80_run_stage6_public7_reproduction.ipynb`](notebooks/80_run_stage6_public7_reproduction.ipynb) is only an optional Colab check against the visible sample wells. See [`docs/stage6_results.md`](docs/stage6_results.md).

For actual model development, run the self-contained [`notebooks/90_run_stage6_pf128_heel_mha.ipynb`](notebooks/90_run_stage6_pf128_heel_mha.ipynb). It keeps Stage 4 and applies a guarded bimodal overlay derived from four stable 16-seed PF batches. The misleading PF128 filename is retained for link continuity; the rejected 128-seed winner-pool configuration is not run. Use its full OOF comparison before building a Kaggle submission Notebook.

For the public-7.x stack improvement, use the attributed and sanitized [`notebooks/95_kaggle_public_mha_safe.ipynb`](notebooks/95_kaggle_public_mha_safe.ipynb). It retains the public 128-seed MHA path while removing leaderboard probes, probe-decoded bias, same-well target transfer, and ambiguous output artifacts. Kaggle import and execution instructions are in [`docs/stage6_public_mha_safe.md`](docs/stage6_public_mha_safe.md).

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
Unmodified downloaded third-party notebooks are kept local. The repository includes only the explicitly attributed, mechanically sanitized Stage 6 derivative needed for reproducible competition execution.
