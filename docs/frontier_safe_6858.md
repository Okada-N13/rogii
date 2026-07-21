# Sanitized 6.858 public frontier

`notebooks/220_kaggle_public_frontier_safe.ipynb` is derived from the user-supplied public notebook reported at 6.858. It is intentionally evaluated as a new baseline before adding Stage 10C.

The build retains the likely transferable changes:

- direction-free midpoint hedge with alpha 2.5, minimum branch separation 2 ft, and a 4 ft cap;
- conservative visible-prefix calibration;
- the original public model, selector, projection, pretrained blend, and hidden-set submission contract.

It removes:

- the global -0.40 ft correction decoded from a leaderboard probe;
- all probe/canary code;
- direct same-well train-target/contact transfer, including dead fallback paths;
- read-only OOF diagnostics and visualizations that lengthen the Kaggle rerun;
- ambiguous submission-shaped CSV files at the end of execution.

The sanitized score is not known in advance. Removing the score-derived global bias may cost roughly 0.01 RMSE if that public estimate remains valid, while the stronger midpoint hedge is expected to retain most of the difference from the earlier 6.997 build.

Import the notebook into Kaggle, attach the same inputs as the successful public MHA run, keep Internet disabled, and run all cells. Submit only `/kaggle/working/submission.csv`. Before submission, verify that `FRONTIER_SAFE_AUDIT` reports matching IDs, finite TVT, and all three removal flags as true.
