# Stage 6 — public 7.x positive control

## Purpose

Stage 6 first closes the implementation gap to a known strong public solution. It runs Kaito Fukami's public `rogii-public-7-061-exact-reproduction` kernel in Colab and saves an audited submission to Drive. Third-party source code is fetched from Kaggle at runtime and is not copied into this repository.

This stage has two deliberately separate score tracks:

- **Public positive control:** the source title reports 7.061; the Kaggle page snapshot saved on 2026-07-19 displayed 7.099. A submitted score in the 7.x range passes this gate.
- **Honest model development:** Stage 4 OOF 12.228593 remains the current leakage-controlled reference. Public leaderboard score, same-well contact override, and visible-prefix self-calibration are not reported as honest OOF.

## Colab execution

1. Download a Kaggle API token and save it as `MyDrive/kaggle/kaggle.json` with the filename unchanged.
2. Confirm that the ROGII competition rules have been accepted on Kaggle.
3. Open `notebooks/80_run_stage6_public7_reproduction.ipynb` in Colab.
4. Run all cells. The long-running public-kernel cell is mainly CPU-bound; a T4 GPU is optional.
5. Upload `MyDrive/kaggle/rogii/submissions/stage6_public7_positive_control_v001/submission.csv` manually.
6. Save the Kaggle public score before changing any components.

The notebook clones the current repository, copies competition data from Drive when necessary, downloads the public kernel and its public artifact datasets, recreates the Kaggle input layout using symbolic links, executes the original kernel, checks submission ID order and finite predictions, and records source/submission SHA-256 hashes.

## Promotion gate

The Stage 6 positive control passes only after Kaggle reports a 7.x public score. It is not promoted as the honest core model. After that result, the public stack must be decomposed under cutback OOF into:

1. 128-seed multi-temperature PF and beam selector;
2. heel GR gain/offset calibration;
3. robust projection;
4. learned trajectory blend;
5. visible-prefix calibration;
6. same-well overlap override.

Overlap-derived target transfer stays isolated from the core model. Components are promoted to the honest pipeline only when they improve pooled OOF and tail metrics without target leakage.
