# Stage 19: lightweight learned trajectory residual

## 目的

安全な実測bestは`6.589`である。Stage 18 ranked retrievalは手元の固定pseudo-testで改善したが、V599推論と組み合わせたKaggle hidden rerunが2回時間超過し、LBで検証できなかった。Stage 19ではdonor探索をKaggleで行わず、学習済みの低次元補正へ圧縮する。

Stage 19AはKaggle提出を作らない。Stage 16B v003の3,865 primary cutsとStage 17 strong-base replayを使うcross-fitted benchmarkである。実測6.589のtop-PFそのもののOOFは存在しないため、Stage 17 replayを代理baseとして用いる。この代理差は結果解釈時に残る主要リスクである。

## モデル

各suffixのbase residualを、prefix境界で滑らかに立ち上がる3項basisへ射影する。

```text
residual(t) = c0*b0(t) + c1*b1(t) + c2*b2(t)
```

学習モデルが予測するのは各cutの`c0,c1,c2`だけである。rowwise neural inferenceやdonor KD-treeは使わない。

特徴はhidden targetを参照しない。

- visible prefixのTVT/U slope
- 全trajectoryのMD/X/Y/Z/GR
- strong-base trajectoryのTVT/U level、slope、curve
- typewell GRとbase trajectoryのshift scan
- typewell signature
- prefix/suffix長とcut fraction

suffixの`TVT`は3係数の教師作成と評価にだけ使用する。8 cutsでsuffix TVTを`+997 ft`しても特徴が完全一致することをgateで確認する。

## 検証

同一wellの複数cutは常に同じfoldへ入る。次の4 familyを独立cross-fitする。

- standard fold
- spatial fold
- typewell fold
- branch-group fold

固定profileはweight `0.50`、cap `16 ft`、ramp `96 rows`。weight/cap gridは診断専用であり、同じOOF上で最良値を選んで昇格させない。

Stage 19Bへのgate:

- standard RMSE `-0.30`以上
- standard bootstrap 95%上限 `< 0`
- 4 familyすべてRMSE改善
- 各familyの80%以上のfoldで改善
- well RMSE P90非悪化
- worst 10% SSE share非悪化
- hidden-target invariance通過

## 実行

Colab Notebook:

`notebooks/480_run_stage19a_trajectory_residual.ipynb`

CPUでよい。通常RAMで不足する場合のみハイメモリを使う。必要artifact:

- `stage16b_testlike_validation_full_v003`
- `stage17_public_replay_full_v002`
- `stage17b_selector_replay_full_v001`

出力run:

`stage19a_trajectory_residual_full_v001`

最後に表示される辞書とprofile診断表を共有する。全gate通過前にKaggle packageやsubmissionを作らない。

## Stage 19A実測

773 wells、3,865 primary cuts、18,841,328 rowsで全gateを通過した。

- fixed profile: weight `0.50`、cap `16 ft`
- standard: `14.5243 → 13.7965`（`-0.7278`）、5/5 folds
- spatial: `-0.5587`、5/6 folds
- typewell: `-0.4627`、5/5 folds
- branch-group: `-0.7443`、5/5 folds
- prefix fractions: 5/5改善
- bootstrap 95%: `[-0.5563, -0.3028]`
- well P90: `-0.7867`
- worst 10% SSE share: `-0.00484`

診断16 profilesはすべてRMSEを改善した。最良診断はweight `0.75`、cap `24 ft`の`-0.8349`だが、同じOOFで選んだ値へ変更せず、事前固定profileでStage 19Bへ進む。

## Stage 19B実測

`notebooks/490_build_stage19b_trajectory_package.ipynb`をCPU Colabで実行し、全gateを通過した。

- 全3,865 cuts、773 wellsで5 seeds × 3係数の15 HGB modelsを学習
- 1 wellあたりの出力は3係数
- raw CSVからの66特徴再計算は最大差`0.0`
- 773 wells benchmarkは`611.915秒`、`0.7916秒/well`
- hidden 200 wellsの追加推定は`158.32秒`
- manifest SHA-256は`5bf1c84bc469395f6cd7042f71dc84f90f287037f35f9ea75a8254997e47d632`

## Stage 19C

実装済み。まずColab CPUで`notebooks/500_build_stage19c_inference_package.ipynb`を単独実行する。

- Stage 19Bのpickleをversion非依存のNumPy NPZへ変換
- 変換前後の予測最大差`1e-12`以下をgate
- Kaggle側はscikit-learn不要、GPU不要、Internet OFF
- 実testの`TVT_input`、MD/X/Y/Z/GR、typewell、既存submissionだけから66特徴を再現
- hidden `TVT`列を一切読まない
- 3係数だけ予測し、固定weight `0.50`、cap `16 ft`で補正

生成される`stage19c_trajectory_inference_package.zip`をKaggle Dataset
`rogii-stage19c-trajectory-inference-package`へアップロードする。その後
`notebooks/510_kaggle_top_pf_stage19_trajectory.ipynb`へ当該Datasetと公開baseに必要なinputを付け、
Internet OFFでinteractive実行する。まず`STAGE19C_TEST_AUDIT`を共有し、全well applied、
`hidden_target_columns_used=False`、追加時間を確認してから提出する。

Stage 19Cは6.589 top-PF baseへ追加する独自学習補正である。代理OOF baseとのずれがあるためLB改善は保証しないが、
Stage 18のdonor探索と異なりhidden 200 wellsでも追加約2.6分の設計であり、時間超過リスクは大幅に低い。
