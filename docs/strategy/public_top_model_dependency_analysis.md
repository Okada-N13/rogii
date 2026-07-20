# ROGII 公開上位 Notebook のモデル依存関係と訓練方法

調査日: 2026-07-20

## 1. 結論

公開上位 Notebook は、すべてを Kaggle Notebook 内で一から訓練しているわけではない。ただし、学習済みモデルを読み込むだけでもない。実態は次のハイブリッド構成である。

1. 学習に時間とメモリを要する GBDT、CatBoost、TCN は Kaggle Dataset から読み込む。
2. test well ごとの Particle Filter、beam search、GR alignment、空間特徴量は Notebook 実行時に作る。
3. 2 系統の学習済み予測を物理系予測と混合する。
4. robust projection、visible-prefix calibration、MHA などを推論後に適用する。

したがって、公開上位の本体は単独の学習済みモデルではなく、`pretrained tabular/sequence model + test-time physics/search + guarded post-process` の複合系である。

## 2. 公開上位と主な依存関係

| Public LB | Notebook | ravaghi artifacts | fleongg models | pilkwang model package | 主な差分 |
|---:|---|:---:|:---:|:---:|---|
| 6.958 | `canqiang/rogii-det-mha160sep4` | yes | yes | no | MHA alpha 1.6 |
| 6.979 | `canqiang/rogii-det-mha140sep4` | yes | yes | no | MHA alpha 1.4 |
| 7.003 | `mitubant/r005-mha120sep4mpkg10-exact-repro` | yes | yes | yes | MHA alpha 1.2 + model-package 最大 1% |
| 7.010 | `chiekhalloul/rogii-det-mha140b-exact-r1` | yes | yes | no | MHA の二峰間隔下限 6 ft |

6.958 と 6.979 のコード上の主要差は MHA の `alpha=1.6` と `alpha=1.4` だけである。0.021 の差は、同一 PF コードで観測されている約 0.09--0.4 RMSE の再実行差より小さい。そのため、MHA 1.6 が真に優れているとはまだ断定できない。

## 3. 外部依存 1: ravaghi/wellbore-geology-prediction-artifacts

Kaggle Dataset: <https://www.kaggle.com/datasets/ravaghi/wellbore-geology-prediction-artifacts>

確認時点では Version 6、合計約 8.04 GB である。上位 Notebook 群の土台になっている。

### 3.1 内容

- `data/train.csv`: 約 7.39 GB、198 列
- LightGBM の serialized `Trainer`: 3 個
- CatBoost の serialized `Trainer`: 2 個
- 各 `Trainer` には学習済み推定器だけでなく OOF prediction、fold score、overall score が含まれる

Notebook は train 特徴量と学習済み Trainer をこの Dataset から読み込む。一方、test 特徴量は実行時に competition data から作る。

### 3.2 学習目的

予測対象は絶対 TVT ではなく、最後の既知 TVT からの差分である。

```text
target = hidden_TVT - last_known_TVT
```

最後の既知 TVT を anchor にするため、井戸ごとの絶対 datum 差をモデルに直接覚えさせにくい。

### 3.3 CV とモデル

- 773 wells を group とする 5-fold `GroupKFold`
- LightGBM 3 個
  - 255 leaves、learning rate 0.03、最大 5,000 trees、seed 123
  - 64 leaves、learning rate 約 0.00934、最大 10,000 trees、seed 0
  - 上と同じ正則化設定、seed 29
- CatBoost 2 個
  - depth 7、最大 8,000 iterations、learning rate 0.02、seed 7
  - depth 7、最大 8,000 iterations、learning rate 0.03、seed 123
- 各モデルは validation RMSE で early stopping
- 5 モデルの OOF prediction を、非負制約付き Ridge で stacking
  - `alpha=1.660283...`
  - `positive=True`

これは row random split ではなく well 単位分割なので、通常の tabular row split より妥当である。

### 3.4 特徴量の性質

単純な X、Y、Z、GR だけではない。Notebook の特徴生成コードには以下が含まれる。

- last-known TVT と既知 prefix の傾き
- Particle Filter の経路と不確実性
- beam-search の複数候補
- typewell GR と horizontal GR の差
- multi-scale NCC alignment
- GR の rolling mean/std、lag/lead、一次・二次差分
- 坑井軌跡の MD、Z、方位、勾配
- ANCC、ASTNU、ASTNL、EGFDU、EGFDL、BUDA の formation surface
- 近隣井戸からの formation-plane KNN と dense ANCC
- 各物理経路どうしの差と分散

したがって、この成果物を「普通の表形式モデル」とだけ呼ぶのは不正確である。PF、beam、地層面、空間情報を tabular model に渡した physics-informed stacking である。

## 4. 外部依存 2: fleongg/rogii-claude-models-pub

Kaggle Dataset: <https://www.kaggle.com/datasets/fleongg/rogii-claude-models-pub>

確認時点では Version 1、約 25.94 MB である。

### 4.1 内容

- `features.json`: 196 features
- `lgb0.pkl`
- `lgb1.pkl`
- `lgb2.pkl`

上位 Notebook では test well から 196 特徴量を動的に構築し、3 個の LightGBM prediction を平均する。この branch が `fleongg_pretrained_submission.csv` の実体である。

### 4.2 訓練方法

公開 Dataset 自体には訓練 manifest がないため、全手順を完全には証明できない。ただし、同じ Notebook に含まれる fallback training code から次が確認できる。

- target は `TVT - last_known_TVT` の delta
- 5-fold GroupKFold、group は well
- 3 個の LightGBM は、255 leaves の強いモデル 1 個と、64 leaves・強正則化の seed 違い 2 個
- early stopping を利用
- package が存在しない場合の完全学習経路には CatBoost 2 個と正の Ridge stack も含まれる

実際に公開 Dataset に入っているのは LightGBM 3 個だけであり、公開上位の deployed branch は CatBoost/Ridge の完全 stack ではなく、3 LightGBM の平均である。

### 4.3 196 特徴量

主なグループは以下である。

- PF-ANCC、PF-Z と標準偏差
- 7 種類の beam path
- 3 window の NCC alignment と信頼度
- 6 formation の full/weighted/recent anchor
- formation-plane KNN、dense ANCC
- GR rolling statistics、lag/lead、gradient
- typewell GR を TVT offset ごとに照合した residual grid
- MD、XYZ、坑井傾斜、eval fraction
- 複数推論器の disagreement

ravaghi 系と独立作者の実装ではあるが、入力信号はかなり重複する。両者の誤差相関が高くなり、単純な blend weight 調整の余地が小さい理由でもある。

## 5. 外部依存 3: pilkwang/rogii-model-package

Kaggle Dataset: <https://www.kaggle.com/datasets/pilkwang/rogii-model-package>

確認時点では Version 9、約 452.16 MB、56 files である。MHA120+MPKG10 系で追加される。

### 5.1 訓練データと検証

公開 manifest により次が確認できる。

- 773 wells
- 3,783,989 rows
- 480 features
- target space: last-known TVT からの delta
- 5-fold GroupKFold
- fold model 20 個と全データ学習 model
- OOF complete rate 100%
- inference では全データ学習 model を使用

formation/KNN feature は query well 自体を除外している。ただし、imputer を fold ごとに完全再構築しておらず、manifest 自身にもその注意が記録されている。空間 CV の評価では追加監査が必要である。

### 5.2 モデル群

- XGBoost
- CatBoost
- HistGradientBoosting
- LightGBM
- sequence TCN

tree 4 系統の単体 OOF RMSE は約 11.02--11.36、TCN は 11.258 だった。単体では公開上位の最終予測より弱いが、誤差の違いを利用して blend する。

### 5.3 TCN

checkpoint と inference code から構造を確認できる。

- 入力: well 内で MD 順に並べた 480 features
- 64 channels
- residual TCN block 6 個
- kernel size 5
- dilation: 1, 2, 4, 8, 16, 32
- 各 block に Conv1d 2 層、GELU、dropout 0.15、residual skip
- 最終 1x1 Conv で row ごとの delta を出力
- checkpoint metadata: best epoch 79

訓練 optimizer、batching、loss の完全な設定は package に含まれていないため、そこは未確認である。

### 5.4 OOF blend

非負 blend の最終 weight は以下である。

| Family | Weight |
|---|---:|
| CatBoost | 0.4496 |
| sequence TCN | 0.4175 |
| HGB | 0.1178 |
| LightGBM | 0.0151 |
| XGBoost | 実質 0 |

- raw blend OOF RMSE: 10.7106
- postprocessed OOF RMSE: 10.6702
- postprocess: delta scale 1.05、Savitzky-Golay window 25

興味深いのは、XGBoost がほぼゼロになり、CatBoost と TCN が大部分を占める点である。同じ 480 特徴量を使っても、sequence model の経路方向の情報が tree model と異なる誤差を持つことが分かる。

### 5.5 上位 Notebook での実際の使用量

MHA120SEP4MPKG10 では、この package prediction を主予測として使わず、base との差に応じて weight を絞る gated correction として使う。最大 weight は 0.01、つまり 1% である。

これは重要である。model package の OOF は整っているが、公開上位 pipeline との prediction scale や誤差構造が異なるため、そのまま大きく混ぜると悪化する。7.003 は package 全体の強さではなく、異なる方向の信号を極小 dose で使った結果と解釈すべきである。

## 6. koolbox-offline の役割

`phongnguyn23021656/koolbox-offline` は学習済み予測モデルではない。Internet OFF の Kaggle 環境で `koolbox.Trainer` と依存 wheel を読み込むための実行環境 Dataset である。

主に以下を提供する。

- serialized Trainer の復元
- GroupKFold training wrapper
- LightGBM/CatBoost と early stopping の統一処理
- OOF prediction、fold score の保持

## 7. Kaggle 実行時に一から計算している部分

外部 Dataset を付けても、次の部分は各 test well に対して毎回実行される。

- 128 seeds x 約 500 particles の likelihood-weighted PF
- likelihood scale 3、5、8、12 の再重み付け
- 複数設定の beam search
- multi-scale NCC
- geometry selector と hold/beam blend
- 近隣 formation-plane / dense ANCC feature
- robust polynomial projection of `TVT + Z`
- visible-prefix cutback test
- MHA の weighted 2-means と posterior-mean shift

したがって、公開モデル Dataset を追加するだけでは score は再現できない。Notebook 内の test-time search と後処理が必要である。

## 8. 最終予測の組み立て

上位 MHA 系を簡略化すると次の通りである。

```text
ravaghi pretrained stack
  + test-time PF / selector / projection
                         -> SP45-side trajectory -- 55% --+
                                                               +-> prefix guard -> MHA
fleongg 3-LightGBM branch --------------------------- 45% --+

optional pilkwang tree+TCN package ---------------- 最大 1%
```

この後に公開版では global bias、same-well contact override、LB probe に関係するセルが含まれる場合がある。現在の safe Notebook ではこれらを削除しているため、公開ページの headline score と完全一致しない可能性がある。

## 9. 6.7--6.8 台の報告との関係

Working Note の 6.794 は、公開 MHA lineage の係数調整とは異なる。

```text
7.080
 -> 6.836: decorrelated neural corrector + robust physics post-process
 -> 6.794: CV で選択した hyperparameter を submission に移植
```

公開 package の分析からも、この方向は合理的である。

- CatBoost と TCN の混合は、同じ特徴量でも単体より OOF を改善した。
- 一方、ravaghi と fleongg は多数の物理特徴を共有し、誤差相関が高い。
- MHA 1.4 と 1.6 の差は seed noise より小さい。
- 次の大きな改善には、同じ PF lineage の weight 調整ではなく、既存 base の残差と相関が低い sequence/physics correction が必要である。

Discussion 上の 6.577 pure-physics、6.798 tabular、5.x model は設計非公開であり、現時点では再現可能な学習レシピとして扱えない。

## 10. 今後の実装方針

公開モデルをそのまま再訓練するより、まず公開 base を固定して残差学習を行う方が効率的である。

1. safe MHA base の train OOF prediction を全 773 wells で固定する。
2. target を `true_TVT - base_OOF_TVT` とする。
3. well GroupKFold と spatial leave-block-out の両方を固定する。
4. 次の候補を小さく試す。
   - 1D TCN residual
   - HGB/CatBoost residual
   - `TVT + Z` の低周波 physics correction
5. 単体 RMSE だけでなく、base residual との相関、well tail、seed 安定性を見る。
6. OOF 上で非負 Ridge または固定小 weight を選ぶ。
7. 同じ correction を全データで再訓練し、Kaggle Dataset として Notebook に追加する。

最初の TCN は 480 特徴量を完全再現する必要はない。既存 Stage 3/4 の特徴に、PF posterior spread、二峰性統計、GR/typewell residual grid、坑井内の位置を追加し、64 channels x 4--6 blocks 程度から始められる。重要なのは TCN 単体の score ではなく、safe public base に重ねた OOF 改善である。

## 11. 未確認事項

- fleongg の 3 個の `.pkl` を作成した元 Notebook と、最終 full-train 手順の完全な manifest
- pilkwang tree model の詳細な optimizer/hyperparameter
- pilkwang TCN の optimizer、loss、batch/window sampling
- 6.675 と報告された Working Note の対応 Notebook
- 6.577 pure-physics および 5.x model の具体的な実装

これらは公開 artifact だけからは確定できない。該当 Notebook、Writeup、Discussion のリンクが得られれば追加調査する。
