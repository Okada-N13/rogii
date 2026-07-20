# Stage 7: 公開モデル残差補正の提出前ゲート

## 目的

Kaggle Notebook を約8時間動かす前に、公開学習済みモデルへの補正が訓練データ上で再現性のある改善かを判定する。実験は Colab、最終推論だけを Kaggle で行う。

現時点の基準提出は Safe MHA140 の Public LB `6.997` である。ただし、Stage 7 が再構築する正規の OOF は、その構成要素である ravaghi 公開スタックであり、Safe MHA 全体の OOF ではない。このため Stage 7 の通過は「補正を MHA に組み込む価値がある」という一次判定であって、`6.997` を必ず上回る保証ではない。

## 検証データ

`ravaghi/wellbore-geology-prediction-artifacts` に含まれる次の資産を使う。

- 公開特徴量付き `data/train.csv`
- 3個の LightGBM Trainer
- 2個の CatBoost Trainer
- 各 Trainer が保存した GroupKFold OOF prediction

5モデルの OOF を公開Notebookと同じ正係数 Ridge で再度 cross-fit し、`alpha=1.0`, `tau_md=85`, `pf_weight=0.09` の後処理を適用する。公開モデルを再学習せず、本物のOOFを使えることが重要である。

## 候補補正

公開特徴、5モデル間の平均・標準偏差・レンジ、公開ベース予測のアンカー差を入力にし、ベース誤差を HistGradientBoosting で予測する。

- 井戸単位 GroupKFold で cross-fit
- 1井戸最大256点に等間隔サンプリング
- 井戸長の違いをサンプル重みで補正
- 2 seed平均
- 残差targetは ±60 ft にclip
- 最終補正は予測残差の25%だけ適用

重み `0.10, 0.20, 0.25, 0.35, 0.50` の結果も診断用に保存する。ただし結果を見て同じOOF上で最良重みを選び、そのまま採用してはいけない。既定の `0.25` が正式なゲート対象である。

## 昇格条件

以下をすべて満たした場合だけ `promoted: true` になる。

1. 通常OOF pooled RMSEが `0.05` 以上改善
2. 通常5 fold中4 fold以上で改善
3. 井戸単位paired bootstrapの95%区間上端が0未満
4. well RMSE P90が悪化しない
5. worst 10% wellsのSSE占有率が悪化しない
6. 地理6 blockで再学習した残差補正が `0.02` 以上改善
7. 地理6 block中5 block以上で改善

空間評価では残差モデルだけを空間blockで再学習している。固定された公開ベースOOFは通常GroupKFold由来なので、この空間テストは保守的な追加監査であり、完全なspatial OOFではない。

## 実行方法

[100_colab_public_residual_gate.ipynb](../../notebooks/100_colab_public_residual_gate.ipynb) を Colab で開き、上から順に実行する。`00_colab_setup.ipynb` は不要で、このNotebookだけで環境構築・公開Dataset取得・検証を完結する。

フル判定では `LIMIT_WELLS = None` のままにする。少数井戸のsmoke結果は提出可否に使わない。

主な出力はGoogle Driveの次の場所に残る。

```text
MyDrive/kaggle/rogii/artifacts/stage7_public_residual_gate_full_v001/
├── gate_summary.json
├── weight_grid.json
├── base_oof.parquet
├── oof.parquet
├── spatial_oof.parquet
├── feature_columns.json
├── public_residual_full_seed_42.pkl
├── public_residual_full_seed_43.pkl
└── model_manifest.json
```

## Kaggle実行へ進む条件

最終セルの辞書で `promoted` が `true` の場合のみ、次にKaggle用推論Notebookへ補正モデルを移植する。`false` の場合は `gates`, `rmse_delta`, `bootstrap_95pct`, `spatial_delta`, `weight_grid` を確認し、Kaggle提出は行わない。

