# Stage 3 residual sequence results

## Outcome

Stage 3 trains a lightweight nonlinear residual model on top of the fixed Stage 2 predictions. The promoted model averages two independently seeded `HistGradientBoostingRegressor` models per fold and applies 60% of their predicted correction.

| Model | Pooled OOF RMSE | Well median | Well P90 | Max well |
|---|---:|---:|---:|---:|
| Stage 2 PF/trend blend | 12.565438 | 7.5120 | 18.2456 | 63.1110 |
| Stage 3 seed 42 | 12.316941 | 7.5674 | 17.2720 | 65.2789 |
| Stage 3 seed 43 | 12.306563 | 7.5685 | 17.1068 | 65.2968 |
| Stage 3 two-seed ensemble | **12.299725** | 7.6068 | **17.2828** | 65.2876 |

The two-seed ensemble improves pooled RMSE by `0.265714`. It improves 463 of 773 wells, and its well-paired bootstrap mean RMSE delta is `-0.212787` with a 95% interval of `[-0.324422, -0.102951]`.

Every fold improves:

| Fold | Stage 2 | Stage 3 | Delta |
|---:|---:|---:|---:|
| 0 | 14.371068 | 14.273797 | -0.097271 |
| 1 | 11.599041 | 11.363522 | -0.235519 |
| 2 | 10.639736 | 10.175380 | -0.464356 |
| 3 | 12.968310 | 12.736559 | -0.231751 |
| 4 | 12.841535 | 12.468930 | -0.372605 |

## Model and validation

The residual target is `TVT_true - Stage2_prediction`. Five models are cross-fitted using the existing well-group folds: a validation well is never used to train the model that predicts it. Each fold trains on at most 256 evenly spaced rows per training well, with sample weights restoring the original pooled-row objective.

The visible-input feature set contains:

- multi-scale centred GR mean and standard-deviation features at windows 9, 33, and 129;
- GR gradients and robust within-well normalization;
- PF surface displacement, slope, seed disagreement, GR scale, and likelihood spread;
- typewell GR mismatch at the Stage 2 predicted TVT;
- trajectory slope, horizon, and prediction length;
- robust surface trends fitted only to the known `TVT_input` prefix at 100, 500, and 1,000 ft windows.

Complete hidden-suffix GR and trajectory values are allowed by the competition setup. Hidden horizontal-well TVT, formation columns, spatial neighbours, and contact overrides are excluded. A regression test adds `+999` to every hidden TVT and verifies that all residual feature columns remain unchanged.

## Correction strength and seed stability

Leaving each fold out while choosing correction strength on the other four selected `0.70, 0.60, 0.60, 0.65, 0.55`. The nested OOF RMSE was `12.311963`; the promoted round weight is `0.60`. The full-OOF optimum was near `0.65`, but the more conservative nested-centre value is used.

Changing the model seed from 42 to 43 changed RMSE by only `0.010378`. Averaging both predictions improved further to `12.299725`. The complete integrated two-seed run took `89.28` seconds on the local CPU reference machine.

## Trade-offs

The model materially improves pooled RMSE and P90, but the median well RMSE rises from `7.5120` to `7.6068`, the maximum well error rises by about 2.18 ft, and the worst-well SSE share increases. It is therefore promoted for the pooled competition metric, while its tail behaviour remains a specific Stage 4 target. Dynamic per-well routing is not introduced because signed correction selection remains less reliable than a fixed ensemble.
