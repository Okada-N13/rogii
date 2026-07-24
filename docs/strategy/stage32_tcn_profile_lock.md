# Stage 32: TCN uncertainty profile lock

## 目的

Stage 31Aで事前定義済みだった4つのuncertainty profileを、保存済み62-cut reportだけで
完全監査する。primary以外にも頑健な候補が存在するかを確認し、予約120 wellsを開く前に
推論profileを1つへ固定する。

これは新しいparameter searchではない。Stage 31A実行後に候補、weight、confidence power、
sign-agreement閾値を追加・変更しない。

## Stage 32A

入力は`stage31a_tcn_uncertainty_full_v001/uncertainty_cut_report.parquet`。
TCNの再学習と再推論は不要で、CPUで数秒から数十秒程度を想定する。

各profileを次の全gateで判定する。

1. pooled RMSE deltaが`-0.10`以下
2. well bootstrap 95% CIの上限が`0`未満
3. cut RMSE P90が非悪化
4. standard foldが4/5以上改善
5. fractionが3/4以上改善
6. spatial foldが4群以上改善
7. typewell foldが4群以上改善
8. branch-group foldが4群以上改善

全gateを通過した候補のうちpooled RMSEが最小のものを1つだけ固定する。通過候補がなければ
TCN uncertainty familyを終了する。Stage 32Aでは予約120 wellsを使用しない。

## Stage 32B

Stage 32Aで候補が固定された場合に限り、未使用の予約120 wellsを一度だけ開封する。
profile、weight、cap、ramp、gateは変更しない。予約確認が通過するまでKaggle提出物を作らない。

## 実行

Colabで`notebooks/690_run_stage32a_tcn_profile_lock.ipynb`を単独実行する。GPUは不要。
