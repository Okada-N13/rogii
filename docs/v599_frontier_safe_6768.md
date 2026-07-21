# V599 A130 public frontier safe notebook

`notebooks/230_kaggle_v599_a130_frontier_safe.ipynb` is derived from the user-supplied public notebook reported at 6.768. It is now the strongest public-model baseline available in this project.

The notebook retains the parts that distinguish that run:

- the `vp_balanced_modelpkg_005` profile and 0.60 SP45 blend;
- balanced visible-prefix calibration with the A130 multiplier;
- the tiny, gated model-package correction capped at 0.00425;
- the conservative PF seed-branch hedge with strength 1.0 and a 3 ft cap, skipping wells already handled by an earlier route.

For a clean and reproducible submission, it removes all paths that read a training well's hidden `TVT` as a test-well correction, disables overlap/contact-target overrides, removes leaderboard-derived bias/probe paths, and leaves exactly one submission-shaped CSV: `/kaggle/working/submission.csv`.

The reported 6.768 belongs to the original public notebook. Sanitization is intentionally narrow, but the safe notebook's exact leaderboard score is not known until it is submitted. Do not add Stage 10C or another correction layer to this first verification run; establish the clean baseline before testing combinations.

## Kaggle execution

Import the notebook into Kaggle and attach the same competition data and public model datasets as the 6.768 source notebook. Keep Internet disabled. A P100 is sufficient and is preferred when only one GPU is actually used by the notebook. Run all cells from a fresh session.

At completion, verify that `V599_FRONTIER_SAFE_AUDIT` reports 14,151 rows, matching ID order, finite TVT, and removal of ambiguous CSV files. Submit only `/kaggle/working/submission.csv`. Save the audit dictionary and the public score for comparison with 6.768, 6.858, 6.994, and 6.997.
