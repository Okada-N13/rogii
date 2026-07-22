# Stage 12A: raw multi-scale NCC benchmark

Stage 11C promoted `w075_cap50` as the only surface profile that remained fold-safe across ordinary, spatial, and typewell holdouts. Stage 12A fixes that profile and asks a narrower question: does raw GR similarity place the true residual TVT offset near the top of a deterministic candidate grid?

## Candidate grid and emissions

For every pseudo-test cut, the Stage 11 OOF coefficients reconstruct the fixed surface trajectory. Stage 12A evaluates 61 constant-offset states from -60 to +60 ft in 2 ft increments. Horizontal and typewell GR are compared using rolling normalized cross-correlation at downsampled windows 5, 13, and 25. The primary emission is fixed before evaluation as:

```text
ncc_mix = 0.40 * NCC(window=13) + 0.60 * NCC(window=25)
```

No suffix TVT enters surface or NCC cost construction. A runtime audit adds 999 ft to hidden suffix TVT and requires byte-identical surfaces and costs.

## Why rank is measured before path RMSE

Repeated GR motifs can make a row-wise raw argmin choose the wrong branch. That does not mean GR is uninformative: if the correct state is regularly in the top 5 or top 10, a learned emission and a path prior can resolve the ambiguity. Stage 12A therefore reports:

- valid typewell/GR coverage;
- median and mean true-state rank;
- Top-5 and Top-10 recall versus random recall;
- offset correlation;
- raw row-wise aligned RMSE;
- quantized/clipped oracle RMSE;
- ordinary, spatial, typewell, fold, and cut consistency.

## Promotion gate

The fixed `ncc_mix` advances to Stage 12B when:

- hidden-target invariance passes;
- valid emission coverage is at least 65%;
- Top-10 recall is at least 1.25 times random in ordinary, spatial, and typewell OOF;
- median true-state rank is at most 25 of 61;
- at least four ordinary folds beat random Top-10 recall;
- the offset-grid oracle improves surface RMSE by at least 2 ft.

Raw aligned RMSE is deliberately not a promotion gate. Stage 12B exists to replace the raw argmin with a learned state distribution and later a deterministic K-best path.

## Colab

Open `notebooks/270_run_stage12a_raw_ncc_benchmark.ipynb`. It reuses Stage 11/11C artifacts and runs on CPU; high RAM is recommended. Keep `LIMIT_WELLS = None` for the decision run. No submission is generated.

The 65% coverage floor was fixed before the full run after a 24-well wiring check showed that about one third of true residual states lie too close to the finite typewell TVT/GR boundary for a complete local NCC window. The same check produced 25.3% Top-10 recall versus 16.4% random, median rank 24 of 61, and a 7.18 ft oracle RMSE. Those values are smoke diagnostics, not promotion evidence.
