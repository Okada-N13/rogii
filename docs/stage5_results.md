# Stage 5 spatial correction audit

## Decision: not promoted

Stage 5 predicts each validation well's Stage 4 residual bias and residual slope from four spatially nearest training wells. Inverse-distance weights are shrunk toward the training-fold global mean over a 10,000 ft distance scale, and only 10% of the capped correction is applied.

The local reference result was:

| Evaluation | Base RMSE | Spatial RMSE | Delta |
|---|---:|---:|---:|
| Standard five well folds | 12.204505 | 12.131975 | **-0.072531** |
| Six leave-spatial-block-out folds | 12.204505 | 12.195102 | **-0.009404** |
| Shuffled neighbour targets | 12.204505 | 12.236989 | **+0.032483** |

The standard-fold result improved all five folds, and the well-paired bootstrap interval was `[-0.069551, -0.023920]`. Shuffling the spatial target association removed the gain and caused degradation, so the ordinary-fold spatial signal is real within this dataset.

However, the gain nearly disappeared when complete geographic regions were held out. Two of six spatial blocks worsened (`+0.0173` and `+0.0297` RMSE), and the worst-10% SSE share increased from `0.564780` to `0.565455` under the standard folds. The model therefore fails the predeclared spatial-block and tail gates and is not added to Stage 4.

This is an incremental spatial-correction audit: Stage 4 predictions remain their existing honest well-OOF values while only the spatial correction is refitted per geographic block. The full Stage 3 residual learner is not retrained under spatial blocks. Consequently, this audit is already a permissive gate for the added spatial component; a component that fails here should not be promoted.

## Search performed

The audit searched neighbour counts `4, 8, 16, 32, 64`, distance shrink scales `500, 1,000, 3,000, 10,000` ft, and inverse-distance powers `0.5, 1, 2`. No tested setting improved both standard and spatial-block pooled RMSE by at least `0.02`. The best fixed-weight spatial-block gain was about `0.010`, while the strongest ordinary-fold settings improved approximately `0.07--0.08`.

This is the failure mode anticipated in the strategy: nearby wells explain residuals when validation wells are interleaved with training wells, but do not extrapolate reliably to an unseen region. Stage 4 remains the promoted model.

## Leakage and controls

- Each validation well is excluded before neighbour target calculation.
- The correction uses only heel `X/Y` and training-well residual summaries.
- Spatial blocks are created by unsupervised K-means on coordinates.
- A regression test changes validation targets by `+999` and verifies unchanged predictions for that validation fold.
- The shuffled-target control uses the same coordinates and model parameters.

The first native Colab Stage 5 audit should be treated as the canonical numerical reproduction because the local prerequisite stack differs slightly from the native Colab Stage 3/4 artifacts.
