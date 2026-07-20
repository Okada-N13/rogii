# Stage 7B: 公開OOFのnested physics gate

Stage 7の残差HGBは、通常5 foldすべて、空間分割、bootstrapのすべてで悪化したため不採用とした。Stage 7Bではモデル容量を増やさず、既知prefixだけから計算できる低自由度の補正を評価する。

## 候補

- 公開予測のlast-known-TVTからの変位を `0.98 / 1.00 / 1.02` 倍
- `U = TVT + Z` を既知prefix末尾160点、320点、全点へrobust一次・二次fit
- physics候補と公開予測の差を `±6 / ±12 ft` にclip
- 補正weight `0.03 / 0.06 / 0.10`
- 境界から `85 / 160 ft` でfade-in

候補の多項式fitには各井戸の可視`TVT_input`だけを使用し、評価suffixの`TVT`は使用しない。

## nested selection

各outer foldの設定は、対象foldを除く井戸だけのRMSEで選ぶ。選択側でbaseより`0.02`以上改善しない場合、そのfoldには補正を適用しない。同じ処理を通常5-foldと地理6-blockで独立に行う。

したがって、表示される`nested_candidate_rmse`は、候補設定の選択にも使われなかった井戸だけを連結した値である。全OOFで選んだ最終推論設定はmanifestへ保存するが、昇格判定にはnested predictionだけを使う。

## 実行

[110_colab_public_physics_gate.ipynb](../../notebooks/110_colab_public_physics_gate.ipynb)をColabで上から実行する。Stage 7の`base_oof.parquet`と`spatial_wells.parquet`を再利用するため、公開Datasetの再ダウンロードと公開Trainerの再読込は不要である。

`promoted: true`の場合だけKaggle推論へ実装する。`false`ならSafe MHA `6.997`は変更しない。

