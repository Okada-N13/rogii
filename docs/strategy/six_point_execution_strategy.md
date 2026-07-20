# ROGII Public LB 6 点台到達の実行戦略

策定日: 2026-07-20

## 1. 目的

第一目標は、ルール上安全で再実行可能な Kaggle Notebook から Public LB `6.xxx` を一度取得することである。

その後は単発の lucky run を成果とせず、OOF、空間 holdout、seed rerun で改善を説明できる pipeline に移行する。

成功を二段階で定義する。

### 短期成功

- committed Kaggle Notebook から `submission.csv` を生成
- Public LB `< 7.000`
- probe、LB-derived bias、同一 well の正解転送を使用しない
- Notebook と入力 Dataset version を記録

### 安定成功

- 同一コードの複数実行または seed 変更で大崩れしない
- honest GroupKFold と spatial holdout の両方で base より改善
- correction の OOF 改善が bootstrap と well-tail 指標でも確認できる

## 2. 現在地

### ローカル系列

| Stage | Pooled OOF RMSE | 状態 |
|---|---:|---|
| Stage 1 | 15.7005 | anchor baseline |
| Stage 2 | 12.5654 | PF / alignment |
| Stage 3 | 12.3393 | residual HGB |
| Stage 4 | 12.2286 | tail/path correction |
| Stage 5 | gate 不通過 | spatial correction は不採用 |

ローカル OOF の井戸集合は Public LB より難しく、絶対値を Public LB と直接比較しない。採用判断には同じ fold 上の差分だけを使う。

### Kaggle 公開系列

- 改善前の公開 Notebook の手元再実行: 7.155
- safe MHA140 の手元再実行: `6.997`（短期目標達成）
- 公開 MHA140SEP4 の表示値: 6.979
- 公開 MHA160SEP4 の表示値: 6.958
- MHA120SEP4+MPKG10 の表示値: 7.003

公開 MHA140 と MHA160 の主要コード差は MHA alpha 1.4 と 1.6 だけであり、表示差 0.021 は既知の PF rerun noise より小さい。手元の MHA140 では対象3井戸中1井戸だけが発火し、そのshiftは既に `+4.0 ft` capに達した。この出力条件ではMHA160も同一になるため、再提出しない。

## 3. 全体方針

6 点台へのルートを二本に分ける。

### Track A: 公開上位の安全な再現

目的は最短で 6.9x に到達することである。

- ravaghi pretrained stack
- fleongg 3-LightGBM branch
- 128-seed likelihood PF
- beam / selector / robust projection
- visible-prefix calibration
- direction-free MHA

を維持し、危険な補正だけを除く。

### Track B: 独自の残差改善

目的は 6.8 以下へ進むための主経路を作ることである。

- Track A の strong base を固定
- base OOF residual を新しい target にする
- decorrelated sequence/tree corrector を cross-fit
- `TVT + Z` の physics post-process を別レイヤーとして検証
- OOF で選んだ小さい weight のみを test に移す

Track A は短期スコア用、Track B は private robustness と追加改善用である。両者を同じ提出の係数違いにはしない。

## 4. Track A: 6.9x を取る最短手順

### A0. safe MHA140 の結果を基準にする

現在計算中の `95_kaggle_public_mha_safe.ipynb` を最初の control とする。

結果ごとの判断は次の通り。

| safe MHA140 score | 判断 |
|---:|---|
| `< 7.000` | 短期目標達成。再現情報を固定して Track B へ進む |
| `7.000--7.080` | pipeline は正常。safe MHA160 を1本だけ試す |
| `7.080--7.200` | 公開版との差が係数以外にある。入力 version、profile、削除セルの影響を監査 |
| `> 7.200` | variant 提出を止め、出力経路と外部 Dataset 解決を再確認 |

### A1. safe MHA160

MHA140 safe builder を一般化し、MHA parameter を config 化する。

```text
alpha     = 1.6
min_mass  = 0.22
sep_low   = 4.0 ft
sep_high  = 40.0 ft
shift_cap = 4.0 ft
```

MHA140 から上記 alpha 以外を変えない。新しい Dataset や post-process を同時に追加しない。

実行前に次を出力する。

- MHA active wells 数
- well ごとの mode mass、mode separation、shift
- MHA140 から変更された rows/wells
- mean/max absolute prediction difference
- final submission hash

MHA160 の Public LB が悪くても、1.4/1.6 の差が seed band 内なら係数探索を続けない。

### A2. model-package は保険候補

MHA120SEP4+MPKG10 は公開 7.003 で、6点台への主候補ではない。ただし MHA160 が失敗した場合の decorrelated control として保持する。

- probe、global bias、contact target transfer は削除
- `pilkwang/rogii-model-package` Version 9 を固定
- package correction は最大 1%
- package と base の差が大きい rows では gate を弱めない

提出優先順位は MHA160 より下とする。

### A3. Track A で行わないこと

- 1.3、1.4、1.5、1.6、1.7 の LB 総当たり
- LB score を見て global offset を逆算
- 同一 well の train TVT/contact を test hidden rows に転送
- sample-specific override
- 公開 CSV を ID だけで無監査 blend
- 複数の補正を一度に追加

## 5. Track B: 6.8 以下を狙う残差学習

### B0. strong-base OOF を作る

最重要作業は、Kaggle test だけで強い公開 pipeline を train wells 上にも同じ条件で動かし、cross-fit base prediction を作ることである。

各 train well で実際の hidden suffix を模擬し、既知 prefix だけを feature generator に渡す。

保存物:

```text
well_id
row_index
fold
true_tvt
last_known_tvt
base_tvt
base_residual
pf paths/scales
pf uncertainty
beam candidates
MHA statistics
geometry/GR/typewell features
```

target は次で固定する。

```text
residual_target = true_tvt - base_tvt
```

base 自体の学習と residual model の validation prediction を混ぜないよう、fold provenance を保持する。

最初の実装は `notebooks/100_colab_public_residual_gate.ipynb` とする。公開artifactに保存された5モデルの正規OOFからravaghi branchを再構築し、残差HGBを通常GroupKFoldと空間block refitの両方で検証する。これはSafe MHA全体ではなく構成branchの一次ゲートであるため、通過後にのみMHAへの組み込みとKaggle最終推論を実装する。

### B1. residual CatBoost/HGB

最初は軽く、解釈しやすい tree corrector を作る。

主な入力:

- Stage 3/4 の既存特徴
- PF scale 3/5/8/12 の差
- PF posterior std、p10/p90、mode mass、mode separation
- beam/PF/learned branch の disagreement
- GR rolling/gradient
- typewell offset residual grid
- `MD since prefix` と eval fraction
- `TVT + Z` のbase slope/curvature

探索する correction weight は `0.10, 0.20, 0.35, 0.50` 程度に限定する。tree model 単体の絶対 RMSE ではなく、base に重ねた OOF RMSE で選ぶ。

### B2. residual TCN

tree corrector の OOF infrastructure が完成してから実装する。

初期構成:

- 32--64 channels
- residual TCN blocks 4--6 個
- kernel size 5
- dilation 1, 2, 4, 8, 16, 32
- dropout 0.10--0.20
- row delta residual output
- well ごとの sequence batching
- masked RMSE または tail-balanced Huber/RMSE

480 features の公開 package を丸ごと複製せず、まず 40--100 個の信頼できる feature で始める。

TCN は standalone score で採用しない。base residual との error correlation が tree correction より低く、blend 後に改善するときだけ採用する。

### B3. physics post-process

残差 model と独立に評価する。

基本状態は、坑井に沿った地層面

```text
U = TVT + Z
```

である。既知 prefix との連続性を固定し、hidden suffix の低周波 drift のみを補正する。

候補:

- normalized MD 上の robust linear/quadratic trend
- degree 2--4 の IRLS projection
- correction cap 1、2、4 ft
- prefix からの fade-in 50--150 ft

公開 Working Note では physics post-process が最大の honest gain `-0.57` と報告されているが、その数値をこちらの改善量として仮定しない。同じ fold 上の paired delta だけで判断する。

### B4. 最終 blend

候補を次の順に積む。

```text
base
 -> base + residual tree
 -> base + residual TCN
 -> base + decorrelated tree/TCN blend
 -> 上記 + physics post-process
```

各追加レイヤーを単独 ablation し、三つ目の correction が既存二つの組合せを壊す場合は採用しない。

## 6. 採用 gate

### 6.1 必須 gate

候補は以下をすべて満たす必要がある。

- hidden target invariance test 合格
- well GroupKFold で pooled RMSE 改善
- 5 folds 中少なくとも 4 folds で改善
- paired well bootstrap 95% CI の上端 `< 0`
- well p90 RMSE を悪化させない、または pooled gain に対して許容可能
- worst 10% SSE share を大きく悪化させない
- spatial-block holdout で改善方向が維持される
- correction cap を外したときだけ良くなるモデルではない
- 3 seeds のうち1 seedだけで改善するモデルではない

### 6.2 目標改善幅

Public LB の 0.02 差は判断材料にしない。ローカルで次を目安にする。

- tree/TCN correction: pooled OOF `-0.15` 以上
- physics post-process: pooled OOF `-0.10` 以上
- full stack: pooled OOF `-0.25` 以上
- spatial-block:少なくとも `-0.05`、かつ極端な block 悪化なし

数値を満たさなくても統計的に明確な改善なら保留候補にできるが、Kaggle提出へは入れない。

## 7. 提出計画

日々の LB 係数探索を避け、意味の異なる候補だけを提出する。

### Submission A: public-safe control

- safe MHA140
- 現在の基準 hash と audit を保存

### Submission B: public-safe reach

- safe MHA160
- A との差は MHA alpha のみ

### Submission C: private-expectation

- safe strong base
- honest OOF を通過した residual tree/TCN
- honest OOF を通過した physics post-process

同じ日に係数違いを多数提出しない。A/B のどちらかが 6 点台に入ったら、短期目標のための LB tuning を止める。

## 8. Kaggle Notebook と Dataset の配置

学習は Colab、最終推論は Kaggle Notebook で行う。

### Colab 側

```text
Google Drive/kaggle/rogii/
  artifacts/
    runs/<run_id>/
    oof/
    models/
    reports/
  datasets/
    <package_name>/
```

Colab では以下を生成する。

- fold models
- all-train inference model
- OOF prediction
- feature column list
- preprocessing statistics
- model manifest
- blend/postprocess config
- metrics and well reports

### Kaggle Dataset 側

Internet OFF で読み込める自己完結 package にする。

```text
metadata/model_manifest.json
features/feature_columns.json
features/build_features.py
models/<model files>
configs/blend.json
configs/postprocess.json
reports/oof_metrics.json
```

Kaggle Notebook は訓練せず、test feature generation、model loading、physics inference、submission audit のみを行う。

## 9. 実装順序

今後は次の順序で実装する。

1. 現在の safe MHA140 の Kaggle score を記録
2. safe builder の MHA config 化
3. standalone safe MHA160 Notebook を生成・テスト
4. MHA140/160 prediction difference report を追加
5. strong-base train OOF generator を実装
6. residual CatBoost/HGB の cross-fit 実験
7. physics projection の独立 OOF sweep
8. residual TCN の cross-fit 実験
9. nonnegative blend と全 gate 評価
10. all-train model package と standalone Kaggle Notebook を生成

一度に複数 Stage を実装せず、各 Stage の Colab 結果を確認してから次へ進む。

## 10. 直近の次アクション

現在の safe MHA140 の score が出るまで新しい Kaggle提出は行わない。結果受領後は次のどちらかを行う。

- `7.000--7.080`: safe MHA160 を実装して提出
- それ以外: score帯に応じて A0 の監査または Track B へ移行

短期の6点台達成に最も近いのは safe MHA160 である。ただし、本格的な6.8以下への改善源は MHA係数ではなく、strong-base OOF 上の decorrelated residual model と physics post-process とする。
