# Stage 27: target-safe spatial surface state

## 判断の背景

Stage 26Aは121個のGR affine path stateを監査したが、最良decoderでもbaseを
`+0.1369`悪化させた。oracle余地は`-3.5085`あった一方、oracle rank中央値は
`59.5/121`、branch-group信号は`3/5`、有効path率は`0.633`である。したがって、
GR costの調整やpath decoderの追加探索は終了する。

Stage 11はMD上の`U=TVT+Z`傾き・曲率、Stage 19はsuffix進捗率上の3係数を学習した。
Stage 27Aはこれらを繰り返さず、地層面をXY座標上の局所平面として扱う。

```text
U(X,Y) = U_anchor + gx * (X-X_anchor) + gy * (Y-Y_anchor)
TVT = U - Z
```

`gx, gy`は既知prefixだけからrobust ridgeで推定する。suffixのXYZは既知なので利用可能
だが、suffix TVTは評価以外では読まない。予測は最後の既知TVTへ厳密にanchorされる。

## Stage 27A

Stage 21Bと同じ62 design-validation cutsだけを使う。Stage 24で確保した120 confirmation
wellsは読み込まず、`reserved_confirmation_used=False`を出力する。

固定primaryは、直近1,200 ftの平面、weight `0.10`、cap `8 ft`、96-row rampである。
600/1,200/2,400 ftと小さなweightの組合せは診断であり、同じ62 cuts上の最良profileへ
事後変更して昇格させない。

昇格には以下をすべて要求する。

- hidden suffix TVTを`+997 ft`しても平面予測がbit-identical
- prefix平面が予測する終点U変化と真の終点U変化の相関が`0.15`以上
- baseを含む固定grid oracleが`0.10 RMSE`以上改善
- 固定primaryが`0.03 RMSE`以上改善しbootstrap上限が0未満
- cut P90非悪化
- standard/typewell/branchで4 groups以上、spatialで4 groups以上、fractionで3 groups以上改善

通過時だけ500 training cutsでfold-safeな空間状態モデルを学習する。不通過なら
prefix-only XY plane continuationを棄却し、120 confirmation wellsは未使用のまま残す。

## 実行

`notebooks/640_run_stage27a_spatial_surface_state.ipynb`をColab CPUで実行する。
GPUは不要。Stage 26Aと同じ既存artifactだけを使い、Kaggle submissionは作らない。

