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

## Colab full結果

Stage 17A v002は全gateを通過した。

- primary eligible cuts: 1,868 / 3,865
- primary eligible rows: 9,432,505 / 18,841,328（50.063%）
- eligible RMSE: last-known `14.738` → public OOF `11.354`（`-3.384`）
- full primary hybrid: `23.096` → `22.118`（`-0.978`）
- eligible改善: 5/5 folds
- full hybrid改善: 5/5 folds
- diagnostic hybrid: `12.727` → `11.867`（`-0.860`）
- target最大差: `0.00046875 ft`（float32 roundoff内）

結論: Ravaghi public OOFは強いが、primary suffix行の49.937%は元prefixが未来情報になるため利用できない。このuncovered短prefixがStage 17Bの対象である。

## Stage 17B

`notebooks/380_run_stage17b_selector_replay.ipynb`で、uncovered cutだけにSP45 likelihood PFをscreenする。

- 8 seeds × 96 particles
- suffixを最大512 tracking stepsへ圧縮し、全行へMD補間
- V599 selectorのscale/hold binを使用
- beam weightは記録するがscreenでは未適用
- uncovered subset、Stage 17A full hybrid、5 foldをすべて評価

PF signalがこの軽量条件でも改善しない場合、full 128-seed PFやbeamへ計算時間を追加しない。

### Stage 17B Colab結果

軽量selector screenは全gateを通過した。

- uncovered primary: 9,408,823 rows（49.937%）
- uncovered RMSE: last-known `29.162` → selector `17.123`（`-12.038`）
- full primary: Stage 17A `22.118` → selector hybrid `14.524`（`-7.594`）
- diagnostic: `11.867` → `11.783`（`-0.084`）
- 5/5 fold改善

ただしfold 3の改善は`-0.119`だけで、他foldの`-5.27`から`-13.93`に比べて不安定である。このため全cutでPF計算量を増やす前にStage 17C gateを行う。

## Stage 17C

`notebooks/390_run_stage17c_selector_gate.ipynb`で、selector gainをtarget-free特徴からwell-isolated cross-fitする。

- target label: cut単位の`baseline RMSE - selector RMSE`
- feature: prefix fraction、suffix length、selector code、scale/hold、tracking ratio、GR sigma、likelihood spread
- target由来のRMSE/SSEはfeatureへ入れない
- fixed threshold `predicted_gain >= 0`
- primary判定後のthreshold変更は禁止
- full-data modelは将来のtest inference用に保存

gateがalways-selectorをpooled、fold、P90、well bootstrapで改善した場合だけresolution/beam auditへ進む。
