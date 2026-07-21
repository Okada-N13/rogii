# Stage 11: multi-cut delta-U surface baseline

Stage 11 starts the independent 5-point track. It does not use a public prediction, pretrained competition model, leaderboard offset, or test-specific contact transfer.

## Validation design

Each training well is converted into four pseudo-test cases with prefix fractions 0.35, 0.50, 0.65, and 0.80. For every cut, only TVT values before the cut may enter features. The suffix TVT is used only to fit and evaluate the training target. A runtime invariance audit adds 999 ft to hidden suffix TVT and requires every feature to remain bit-identical while the target changes.

The same fixed experiment is cross-fitted three ways:

- ordinary five-fold well holdout;
- six XY geographic blocks;
- typewell-signature blocks formed from GR distribution and TVT-range descriptors.

The typewell split is a geological-regime stress test, not a claim that multiple horizontal wells share the same physical typewell file.

## Model

The model works in `U = TVT + Z` space. The known prefix provides an exact anchor and a robust local slope. The HGB model predicts only two low-frequency corrections:

```text
U(h) = U_anchor
     + prefix_slope * h
     + learned_slope_correction * h
     + learned_curvature * h^2
```

where `h` is MD distance from the cut in thousands of feet. Inputs include the visible prefix, complete well geometry, complete GR summaries, typewell signatures, and fold-local regional estimates. Regional features for a training cut exclude its own well; validation wells can use only training-fold donors.

Predicted slope, curvature, and final surface movement are capped before evaluation. The learned correction has a fixed predeclared shrinkage weight of 0.35. A full-OOF weight grid is saved only as a diagnostic for the next experiment; it cannot retroactively promote this run. At most 1,024 evenly spaced suffix rows per cut are retained so the three audits fit comfortably in Colab memory.

## Promotion gate

`promoted_to_alignment_benchmark` is true only when all conditions pass:

- hidden-target invariance;
- ordinary pooled RMSE improves by at least 0.10;
- at least four of five ordinary folds improve;
- spatial and typewell holdouts both improve;
- paired-well bootstrap 95% upper bound is below zero;
- well P90 and worst-10% SSE concentration do not worsen.

Promotion authorizes Stage 12 alignment work. It does not authorize a Kaggle submission and does not imply a 5-point leaderboard score.

## Colab

Open `notebooks/250_run_stage11_multicut_delta_u.ipynb`. It is standalone, mounts Drive, clones or updates the repository, installs the locked environment, and writes the run to:

```text
MyDrive/kaggle/rogii/artifacts/stage11_multicut_delta_u_full_v001
```

A GPU is unused. Select a CPU runtime; high RAM is useful. Leave `LIMIT_WELLS = None` for the decision run. `LIMIT_WELLS = 24` is only a wiring check.

Return the complete summary dictionary and fold-delta table. The response determines whether Stage 12 should begin with raw NCC benchmarking or whether Stage 11 needs a target/prior correction first.

## Pre-release smoke check

A 24-well real-data wiring run completed before release. With the fixed 0.35 shrinkage, its ordinary, spatial, and typewell deltas were `-1.284`, `-0.947`, and `-0.177` respectively. Only three of five ordinary folds improved and the bootstrap interval crossed zero, so this is evidence that the pipeline has a usable signal, not a promotion result. The 773-well Colab run remains the first valid decision.
