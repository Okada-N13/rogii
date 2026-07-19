# Stage 6 — public 7.x positive control

## Purpose

Stage 6 first closes the implementation gap to a known strong public solution. Formal scoring is performed by copying and committing Kaito Fukami's public `rogii-public-7-061-exact-reproduction` kernel on Kaggle. Kaggle then reruns it against the hidden test set. Direct CSV upload and a CSV produced in Colab cannot be used for this Code Competition.

This stage has two deliberately separate score tracks:

- **Public positive control:** the source title reports 7.061; the Kaggle page snapshot saved on 2026-07-19 displayed 7.099. A submitted score in the 7.x range passes this gate.
- **Honest model development:** Stage 4 OOF 12.228593 remains the current leakage-controlled reference. Public leaderboard score, same-well contact override, and visible-prefix self-calibration are not reported as honest OOF.

## Formal Kaggle execution and submission

1. Open `https://www.kaggle.com/code/kaitofukami/rogii-public-7-061-exact-reproduction`.
2. Click **Copy & Edit**. The copied Notebook should retain the competition and public dataset Inputs.
3. Confirm that Internet is off and `/kaggle/input/competitions/rogii-wellbore-geology-prediction` is available.
4. Confirm that the artifact Inputs referenced by the Notebook are attached: `wellbore-geology-prediction-artifacts`, `koolbox-offline`, `rogii-claude-models-pub`, and `rogii-model-package`.
5. Click **Save Version**, choose **Save & Run All**, and wait for the committed run to finish.
6. Open the committed version's Output, select `/kaggle/working/submission.csv`, and click **Submit to Competition**.
7. Kaggle starts a separate rerun on the hidden test set. Wait for the submission status and record its public score.

`kaggle.json` is not required for this Kaggle-side execution. The competition allows CPU or GPU Notebook runs up to nine hours and requires Internet to be disabled.

## Optional Colab visible-sample check

1. Download a Kaggle API token and save it as `MyDrive/kaggle/kaggle.json` with the filename unchanged.
2. Confirm that the ROGII competition rules have been accepted on Kaggle.
3. Open `notebooks/80_run_stage6_public7_reproduction.ipynb` in Colab.
4. Run all cells. The long-running public-kernel cell is mainly CPU-bound; a T4 GPU is optional.
5. Inspect the output under `MyDrive/kaggle/rogii/visible_sample_checks/stage6_public7_v001/`.

The Colab helper runs only against the three visible sample wells. Its CSV is not a valid Code Competition submission and does not produce a leaderboard score. It exists only to inspect dependencies, runtime, output format, and source/submission SHA-256 hashes.

## Promotion gate

The Stage 6 positive control passes only after Kaggle reports a 7.x public score. It is not promoted as the honest core model. After that result, the public stack must be decomposed under cutback OOF into:

1. 128-seed multi-temperature PF and beam selector;
2. heel GR gain/offset calibration;
3. robust projection;
4. learned trajectory blend;
5. visible-prefix calibration;
6. same-well overlap override.

Overlap-derived target transfer stays isolated from the core model. Components are promoted to the honest pipeline only when they improve pooled OOF and tail metrics without target leakage.
