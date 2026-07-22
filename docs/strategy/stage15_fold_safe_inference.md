# Stage 15: OOF対応の独立test推論

## 目的

Stage 14Bで検証を通過した `generic_w080_cap16` をcompetition testへ移す。ただしtestの3坑井IDはtrainにも存在するため、同じ坑井の完全なtrain TVTを使った全データ学習は行わない。

各test坑井ではStage 11のfold assignmentを引き継ぎ、次をすべて同じheld-out foldに揃える。

- delta-U surface: 当該foldを除外して再学習
- learned emission TCN: 当該坑井がvalidationだったcheckpointを選択
- residual HGB: 当該坑井がvalidationだったcheckpointを選択

これによりtestの既知prefix、軌跡、GR、typewellだけから推論し、同一坑井のhidden suffix TVTを学習側から遮断する。

## Colab: package作成

1. [330_build_stage15_fold_safe_package.ipynb](../../notebooks/330_build_stage15_fold_safe_package.ipynb) をColabで開く。
2. ランタイムはCPUでよい。
3. 上から全セルを実行する。
4. 最後の結果で `package_ready: true` と `same_well_target_leakage_guard: true` を確認する。

出力先は次の2つ（非公開test対応のv002）。

- Drive上のDataset用フォルダ: `artifacts/stage15_fold_safe_package_v002/package`
- 保管用zip: `artifacts/stage15_fold_safe_package_v002/stage15_inference_package.zip`

Kaggle Datasetには`package`フォルダの内容を登録するのが基本だが、保管用zipをそのまま登録してもよい。Notebookは`manifest.json`を直接検出できない場合、`stage15_inference_package.zip`を自動検出して`/kaggle/working`へ展開する。

## Kaggle: Internet-OFF推論

1. 新しいKaggle Notebookを作る。
2. Competition dataと上記Stage 15 package DatasetをAdd Inputする。
3. InternetをOFFにする。
4. Acceleratorは **T4 x2** を選ぶ。単一GPUしか使用しないが、現在のKaggle PyTorchビルドではP100が`no kernel image`になるため使用しない。T4を選べない場合はCPU推論へ切り替える。
5. [340_kaggle_stage15_internet_off_inference.ipynb](../../notebooks/340_kaggle_stage15_internet_off_inference.ipynb) のセルをコピーし、上から実行する。
6. 最後のauditで次を確認する。

```text
rows: 14151
id_order_matches_sample: True
finite_tvt: True
same_well_target_leakage_guard: True
```

提出対象は `/kaggle/working/submission.csv` の1ファイルだけ。secondary予測は監査値にのみ使い、別CSVを出さない。

## 推論経路

primary提出は次の固定仕様で作る。

```text
fold-safe delta-U surface (weight 0.75, cap 50 ft)
  -> fold-safe learned emission expected offset
  -> fold-safe generic residual HGB (weight 0.80, cap 16 ft)
```

Stage 14Bのnested standard値は11.8047だが、test推論の固定仕様はstandard・spatial・typewell全てで安全だった共通仕様を使う。このCV値からLeaderboardの絶対値は保証できないため、最初の提出はStage 15のtest移植監査でもある。

## packageの安全性

- package構築時にStage 14B promotionと固定profile名を検証する。
- 全収録ファイルのSHA-256を保存し、Kaggle推論開始時に再検証する。
- `TVT_input` は有限値が連続prefixであることを検証する。
- test horizontalにhidden `TVT` 列が渡された場合は停止する。
- trainにfold assignmentがあるtest IDは対応するheld-out foldだけを使い、trainにも存在しない非公開testの未知IDは全fold-safeモデルの平均へ自動的に切り替える。
- sample submissionへone-to-oneで結合し、順序、欠損、有限値を検証する。
- 出力ディレクトリには提出候補CSVを複数作らない。
