# Stage 7F: Kaggle Internet-OFF public model-package blend

Stage 7Eは`package_postprocessed`をweight `0.4`でravaghi branchへ混ぜる条件を採用した。通常GroupKFoldは5/5、空間blockは6/6で改善し、paired bootstrap、well P90、worst 10% shareを含む全gateを通過した。

[150_kaggle_public_robust_blend.ipynb](../notebooks/150_kaggle_public_robust_blend.ipynb)は、この固定条件を提出用のSafe MHAへ組み込む。

変更するのはravaghi branch (`sub_1`) のみである。

```text
sub_1_stage7f = 0.60 * sub_1_ravaghi + 0.40 * package_postprocessed
SP45           = 0.30 * sub_1_stage7f + 0.70 * selector
pre-MHA final  = 0.55 * SP45 + 0.45 * fleongg
```

したがって、MHA overlay前の最終予測に対するpackageの実効weightは`0.40 × 0.30 × 0.55 = 0.066`である。Safe MHAの最終出力へpackageを40%直接混ぜる変更ではない。

## Kaggle Notebook設定

- Accelerator: GPU（P100を優先、なければT4）
- Internet: OFF
- Competition data: `rogii-wellbore-geology-prediction`
- 追加Dataset: `pilkwang/rogii-model-package`
- 既存Safe MHAが必要とするDatasetもすべて追加する

ノートブックはpackage内のmanifest、480特徴量builder、all-train CatBoost/HGB/LGB/TCN、blend configを読み、deltaのalpha補正とwell単位Savitzky–Golay平滑化までpackage定義通りに実行する。TCNだけは公開再現コードと同じくCPU推論として、Kaggle CUDAの互換性エラーを避ける。

実行後は`/kaggle/working/submission.csv`だけを提出する。`stage7f_public_blend_audit.json`にはpackageの検出先、weight、行数、ravaghiとの差分RMSEを保存する。最終セルはID順序、有限値、曖昧なsubmission CSVの除去、SHA-256を再検査する。

この提出はStage 7EのOOF改善がSafe MHA全体でも同じ比率で再現することを保証しない。最初のKaggle提出では6.997のSafe MHAをcontrolとして残し、Stage 7Fとの差だけを比較する。
