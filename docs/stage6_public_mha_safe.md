# Stage 6 public MHA safe submission Notebook

## What this is

`notebooks/95_kaggle_public_mha_safe.ipynb` is an attributed derivative of Can Qiang's public `rogii-det-mha140sep4` Notebook. It is the public-7.x stack plus the reported midpoint hedge, prepared for Kaggle execution.

The build keeps:

- 128-seed likelihood-weighted particle filtering;
- the SP45 / learned-model blend;
- visible-prefix backtesting and conservative calibration;
- direction-free midpoint hedge for credible bimodal PF wells.

The build removes:

- the global `-0.40 ft` correction decoded from a hidden-public canary;
- every leaderboard probe/canary path;
- same-well contact target transfer;
- disabled or read-only smoother and diagnostic experiments.

The final cell validates row count, ID order, and finite predictions, removes other submission-shaped CSV files from `/kaggle/working`, and writes `stage6_public_mha_safe_audit.json`. The only formal candidate left is `submission.csv`.

## Import into Kaggle

The legacy `kaggle.json` in this environment can read public kernels but Kaggle now rejects it for kernel upload, so use the Kaggle UI:

1. Download `notebooks/95_kaggle_public_mha_safe.ipynb` from GitHub.
2. On Kaggle, choose **Create → New Notebook**, then import/upload the downloaded `.ipynb`.
3. Add the competition Input `rogii-wellbore-geology-prediction`.
4. Add these three dataset Inputs:
   - `phongnguyn23021656/koolbox-offline`
   - `fleongg/rogii-claude-models-pub`
   - `ravaghi/wellbore-geology-prediction-artifacts`
5. Set Accelerator to GPU and Internet to Off.
6. Choose **Save Version → Save & Run All**. Do not run cells interactively and submit that draft; the competition scores a committed run against hidden test data.
7. In the committed version, check the last cell. It must show `id_order_matches_sample: true`, `finite_tvt: true`, `probe_bias_removed: true`, and `lb_probe_removed: true`.
8. From Output, select `/kaggle/working/submission.csv` and submit it to the competition.

The MHA source reported roughly `-0.16` on its fork, but the sanitized Notebook has not yet received its own Kaggle score. Removing the probe-decoded global bias can also move the score slightly. Therefore this is a plausible improvement over the 7.099 positive control, not a guaranteed score.

## Rebuild and audit

The derived Notebook is generated mechanically:

```bash
python scripts/build_public_mha_safe_notebook.py \
  --source public_notebook/rogii-det-mha140sep4.ipynb \
  --output notebooks/95_kaggle_public_mha_safe.ipynb
```

Automated tests fail if the MHA cell is missing or any probe, canary bias, contact override, ambiguous-artifact handling, or attribution marker regresses.
