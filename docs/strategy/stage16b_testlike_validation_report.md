# Stage 16B: test-like branch-overlap validation結果

実施日: 2026-07-22  
ローカルfull run: `artifacts/stage16b_full_local_v001`

## 固定した条件

- train wells: 773
- primary prefix fractions: `0.18, 0.22, 0.26, 0.30, 0.34`
- diagnostic fractions: `0.40, 0.50, 0.65`
- primary cuts: 3,865
- all cuts: 6,184
- 採点行: 26,225,067 suffix rows
- donor graph: 95,752 target-free edges
- branch groups: 682
- hidden-target invariance: pass
- manifest SHA-256: `0d85e3e2842eb635a2a9231ee5086923166bcd0fa6d49d6ad87f871099968312`

branch groupは610 singleton、72 multi-well groupsで、163 wellsがmulti-well groupに属した。最大group sizeは6だった。

## Control結果

| Control | 全cut RMSE | primary RMSE | diagnostic RMSE | 判断 |
|---|---:|---:|---:|---|
| last-known TVT | **20.708** | **23.096** | **12.727** | 凍結control |
| constant-U | 96.092 | 103.911 | 72.409 | 不採用 |
| linear-U | 52.327 | 59.808 | 24.447 | 不採用 |

last-known TVTのprefix fraction別RMSE:

| Fraction | RMSE |
|---:|---:|
| 0.18 | 36.478 |
| 0.22 | 21.670 |
| 0.26 | 16.876 |
| 0.30 | 15.412 |
| 0.34 | 14.523 |
| 0.40 | 13.416 |
| 0.50 | 12.646 |
| 0.65 | 11.577 |

## 解釈

旧Stage 11--14の主cut `0.35--0.80`は、短prefix testを大幅に過小評価していた。last-known TVTでもprimary RMSEは23.096で、diagnostic帯の12.727から約10.37悪化する。特に0.18 cutは36.478である。

Stage 15の35.110は、短prefix条件で不安定なconstant/linear-U系surfaceを主経路にした結果として説明可能である。旧CVで良かったdelta-U surfaceを調整して再提出する方針は棄却する。

## 決定

- Stage 16B manifestとfoldを以後固定する。
- 主昇格指標はprimary cutsの全suffix pooled RMSEとする。
- `last_tvt`を最小controlにする。
- constant-U/linear-Uをstrong baseとして使わない。
- Stage 17でV599/public strong-baseを同じmanifestへreplayする。

## 再現Notebook

`notebooks/360_run_stage16b_testlike_validation.ipynb`

Colab full runのsummaryで次を照合する。

```text
n_wells: 773
n_cuts: 6184
n_primary_cuts: 3865
hidden_target_invariance: True
manifest_sha256: 0d85e3e2842eb635a2a9231ee5086923166bcd0fa6d49d6ad87f871099968312
```
