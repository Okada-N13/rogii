# Stage 4 submission workflow

## Recommended submission

Run `notebooks/70_generate_submission.ipynb`. The notebook is self-contained and reuses existing Stage 2 and Stage 3 artifacts. It writes:

```text
MyDrive/kaggle/rogii/submissions/stage4_submission_v001/
├── submission.csv
├── submission_stage3.csv
├── submission_report.json
├── test_predictions.parquet
├── config.yaml
└── environment.json
```

Upload `submission.csv` as the primary submission. It uses the promoted Stage 4 pipeline. `submission_stage3.csv` is a secondary control that omits the uncertainty guard and trellis.

## Prediction path

1. Stage 2 runs the two-batch PF/trend blend on each test horizontal well.
2. The two Stage 3 full models predict residual corrections from visible sequence features.
3. Stage 4 caps the correction for test wells above the training OOF uncertainty threshold.
4. The deterministic GR/typewell trellis contributes its fixed 10% correction.
5. Predictions are joined to the official sample submission by exact ID.

Stage 5 spatial correction is excluded because it failed geographic-block and tail gates. The submission code also excludes train/test same-well target copying and the public contact override. Train targets are not opened during test prediction.

## Mandatory validation

The generator refuses to finish unless all of the following hold:

- columns are exactly `id,tvt`;
- predicted IDs exactly equal sample-submission IDs;
- output order equals the sample order;
- no duplicate IDs;
- no missing or infinite TVT values.

The local test snapshot produced 14,151 rows across three wells with zero missing or duplicate IDs. Stage 3 and Stage 4 test predictions differed by approximately `0.66` ft RMSE and had correlation approximately `0.999998`, so they are controls rather than strongly diversified submissions.

## Kaggle upload

Open the competition page, choose **Submit Predictions**, and upload:

```text
MyDrive/kaggle/rogii/submissions/stage4_submission_v001/submission.csv
```

Suggested description:

```text
Stage4 honest PF + residual ensemble + uncertainty guard + trellis
```

Do not choose between Stage 3 and Stage 4 solely from a small public-leaderboard difference. Stage 4 remains primary because it improved honest OOF, every fold, well median/P90, and the paired-bootstrap interval.
