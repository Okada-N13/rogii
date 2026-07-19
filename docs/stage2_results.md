# Stage 2 particle-filter results

## Outcome

Stage 2 adds a leakage-controlled multi-seed particle filter (PF) that tracks the smooth geological surface `TVT + Z`. The promoted model is a fixed 75% two-batch PF + 25% Stage 1 guarded-trend blend.

| Model | Pooled OOF RMSE | Well median | Well P90 | Max well |
|---|---:|---:|---:|---:|
| Anchor baseline | 15.909853 | 10.6651 | 22.9725 | 70.6394 |
| Stage 1 trend blend | 15.700508 | 10.5695 | 22.9146 | 67.8831 |
| Stage 2 PF, seed 42 | 13.178118 | 7.8995 | 20.5276 | 60.0341 |
| Stage 2 PF, seed 43 | 13.930864 | 7.8244 | 21.1257 | 69.3106 |
| Stage 2 two-batch PF average | 12.899535 | 7.6841 | 19.4894 | 61.6112 |
| Stage 2 promoted 75/25 blend | **12.565438** | **7.5120** | **18.2456** | 63.1110 |

The promoted blend improves Stage 1 by 3.135070 pooled RMSE and improves every fold:

| Fold | Stage 1 | Stage 2 blend | Delta |
|---:|---:|---:|---:|
| 0 | 17.468689 | 14.371068 | -3.097621 |
| 1 | 14.367166 | 11.599041 | -2.768125 |
| 2 | 15.612290 | 10.639736 | -4.972554 |
| 3 | 16.984718 | 12.968310 | -4.016408 |
| 4 | 13.715257 | 12.841535 | -0.873722 |

## Blend selection

The PF weight was searched only on the fixed OOF predictions. To estimate selection bias, each validation fold was held out while the other four folds selected from weights in increments of 0.05. For the two-batch PF, the selected weights were `0.75, 0.80, 0.75, 0.75, 0.80`, and the resulting nested OOF RMSE was `12.597170`. We therefore use the round, conservative PF weight `0.75`, rather than a more precise whole-OOF optimum.

A well-paired bootstrap with 2,000 resamples gave:

- versus Stage 1: mean well-RMSE delta `-2.9686`, 95% interval `[-3.3877, -2.5380]`;
- versus the two-batch PF alone: mean well-RMSE delta `-0.1484`, 95% interval `[-0.2921, -0.0100]`.

## Leakage controls

For each horizontal well, the PF receives only:

- the known `TVT_input` prefix and its final surface anchor;
- the complete visible `MD`, `Z`, and `GR` sequence;
- the paired typewell's `TVT` and `GR` profile.

It does not use the horizontal well's hidden target `TVT`, formation columns, neighbouring-well labels, or the public sample contact override. A synthetic regression test changes every hidden target by `+999` and verifies byte-equivalent predictions. Typewell pairing is derived from the horizontal filename and validated before prediction.

## Implementation and runtime

The PF uses 256 particles per seed, two independent batches of 16 deterministic seeds, state variables for surface position and rate, systematic resampling, and a deliberately weak GR likelihood. The single-batch RMSE changed from `13.178118` at seed 42 to `13.930864` at seed 43, so a single favourable batch was not promoted. Averaging the batches reduced the PF RMSE to `12.899535`. Its OOF output also stores total seed/batch disagreement, estimated GR scale, and the log-likelihood spread for later tail analysis and Stage 3 residual modelling.

The full 773-well seed-42 and seed-43 runs took `905.62` and `907.05` seconds on the local CPU reference machine. The promoted two-batch run therefore needs about 30 minutes locally; allow 30--45 minutes in Colab. The eight-well promoted smoke run completed successfully with RMSE `11.146564`.

Use `notebooks/30_run_stage2_pf.ipynb` in Colab after `00_colab_setup.ipynb`. The complete run uses `configs/experiment/pf_trend_blend.yaml` and persists results to Google Drive.

## Next experiments

The first priority is not blindly increasing PF compute. The larger gains expected in Stage 3 are:

1. heel-adapted pseudo-typewell profiles;
2. better robust window/gradient likelihood evidence;
3. a small 1D CNN or BiGRU trained on honest PF residuals;
4. structural projection and a small decorrelated path component.

All candidates should continue to use fixed well folds, hidden-target invariance tests, seed reruns, and full-stack OOF comparisons.
