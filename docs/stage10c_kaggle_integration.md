# Stage 10C: Kaggle public MHA integration

Stage 10C integrates the OOF-promoted `prefix030_cap8` alignment into the verified 6.997 public MHA notebook. The new cell runs immediately after the ravaghi `sub_1` prediction and before the 0.3 ravaghi / 0.7 selector blend.

The profile is fixed before test inference:

- branch weight: 0.20;
- raw correction cap: 8 ft;
- minimum visible-prefix GR-shape correlation: 0.30;
- loose multi-scale alignment with candidate offsets from -20 to +20 ft.

Only test horizontal GR, visible `TVT_input`, typewell GR/TVT, and the ravaghi prediction are read. Hidden TVT targets, leaderboard probes, contact-target transfer, and internet downloads are not used. The subsequent MHA, projection, pretrained blend, visible-prefix layer, midpoint hedge, and final single-file audit are preserved byte-for-byte from notebook 95.

## Kaggle execution

Import `notebooks/210_kaggle_public_mha_stage10c.ipynb` as a new Kaggle notebook. Attach exactly the same competition data and model datasets used by the successful 6.997 run. Keep Internet disabled and use the same accelerator as that run. Execute all cells and submit only `/kaggle/working/submission.csv`.

Before submitting, retain these two printed dictionaries:

- `STAGE10C_ALIGNMENT_AUDIT`, which reports active wells and branch movement;
- the final safe audit, which verifies row count, ID order, finite values, hash, and removal of ambiguous submission-shaped CSV files.

This is a deliberately conservative integration. Because ravaghi is later weighted by 0.3 and the SP45 branch by 0.55, the nominal linear effect before projection is `0.20 × 0.30 × 0.55 = 0.033` of the raw alignment correction.
