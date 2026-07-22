# Stage 17A: public strong-baseのtest-like replay

## 目的

実測LB 6.685のV599 pipeline全体を偽のCVへ変換せず、provenanceが確認できる公開GroupKFold OOF branchからStage 16B manifestへ移す。

## 使用する予測

`stage7_public_residual_gate_full_v001/base_oof.parquet`のRavaghi public stackを使う。これは5つの公開booster OOFをpositive Ridgeでcross-fitし、公開Notebookと同じdelta後処理を行った予測である。同一wellは学習foldから除外されている。

各wellの公開OOFは、competition trainに元から設定されたknown prefixで生成されている。疑似cut `c` に対して元prefix終端 `o` が `o <= c` の場合だけ、公開予測が見た情報は疑似prefixの部分集合になる。この場合だけreplayを有効とする。

```text
replay_eligible = original_public_cut_index <= pseudo_cut_index
```

`o > c`は未来のTVT_inputを見ているため使用しない。未eligible cutはfull hybrid指標ではlast-known TVTのままにする。

## 出力

- `replay_predictions.parquet`: eligible suffix行だけのtrue TVT、public OOF、last-known control、residual target
- `cut_report.parquet`: 全6,184 cutのcoverage、SSE、fold、eligibility
- `summary.json`: primary/diagnostic coverage、eligible改善、full hybrid改善、fold consistency
- `provenance.json`: Stage 16B、public OOF fold、target alignmentの監査

## 昇格条件

- Stage 16B manifest hash一致
- public OOFがwell単位5-fold
- raw TVTとのtarget alignment一致
- primary suffix行coverage 35%以上
- eligible subsetでRMSE 0.05以上改善
- Stage 16 standard foldの4/5以上でeligible改善
- 未coveredをlast-knownとしたfull primary hybridも改善

## まだ検証しないV599成分

- SP45 selector PF
- projection
- learned trajectory branch
- visible-prefix calibration
- model-package correction
- PF seed-branch hedge

これらをRavaghi OOFの改善値へ混入させない。Stage 17A通過後、計算量を限定したStage 17BでSP45 selectorを短prefix未covered cutへreplayする。

## 実行

`notebooks/370_run_stage17_public_replay.ipynb`をCPU Colabで単独実行する。Stage 16B v003と以前のStage 7 `base_oof.parquet`がDriveに必要である。Kaggle提出は行わない。
