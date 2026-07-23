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
