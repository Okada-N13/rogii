# Stage 10A: multi-scale GR/typewell alignment

## Purpose

Stage 9 showed that the residual TCN output was too small and unstable across folds. Stage 10 changes the target from direct residual regression to a constrained structural alignment problem. The verified ravaghi OOF prediction remains the base; the new model is allowed only to move it along the TVT axis when observed horizontal-well GR and typewell GR support that move.

## Method

For every well, the method calibrates typewell GR to the visible prefix, constructs candidate TVT offsets from -20 to +20 ft, and scores each offset by normalized GR-shape correlation at three window sizes. A Viterbi path enforces continuity, penalizes large offsets, and limits abrupt changes. Three stiffness profiles are evaluated.

The resulting correction is not accepted directly. Every `(profile, blend weight)` pair is selected with nested standard folds. A pair is eligible only when it improves the inner-fold pooled RMSE by at least 0.02 and does not worsen any inner fold. A single inference specification must also pass the same all-fold robustness condition.

## Promotion rule

Stage 10A is an OOF audit, not a Kaggle submission builder. It advances only when all of the following hold:

- pooled RMSE improves by at least 0.05;
- at least four of five standard folds improve;
- the paired well-bootstrap upper 95% bound is below zero;
- well p90 and worst-10% SSE concentration do not worsen;
- one fixed branch and weight is robust across all five folds.

If `promoted_to_spatial_audit` is false, this line is closed without a Kaggle run. If true, the next step is a separate spatial cross-fit audit before any public-notebook integration.

## Colab

Open `notebooks/190_run_stage10_multiscale_alignment.ipynb` in Colab and run all cells. A CPU high-RAM runtime is preferred; the algorithm does not use a GPU. The notebook clones or updates the repository and installs the environment itself. It reuses the Stage 8 matrix on Google Drive, or rebuilds it from the persisted Stage 7 base when necessary.
