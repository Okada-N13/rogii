# Stage 16A: 6.685 / 6.693 frontier差分監査

実施日: 2026-07-22

## 結果

`notebooks/230_kaggle_v599_a130_frontier_safe.ipynb`と`notebooks/240_kaggle_branch_overlap_6594_safe.ipynb`は、ともに51セルである。差があるのは次だけだった。

| Cell | 種類 | 差分 |
|---:|---|---|
| 0 | Markdown | タイトルと説明 |
| 48 | Code | 最終PF seed-branch midpoint hedge |
| 50 | Code | profile名・監査ファイル名・監査値 |

Cell 1--47の全code、入力dependency、model探索、V599、SP45 blend、A130 visible-prefix calibration、model-package branch、PF生成は同一である。

## 唯一の予測差

| 設定 | 6.685版 | 6.693版 |
|---|---:|---:|
| hedge strength | 1.00 | 0.60 |
| hedge cap | 3 ft | 2 ft |
| minimum minor mass | 0.25 | 0.25 |
| separation | 4--40 ft | 4--40 ft |
| prior route済みwell | skip | hedgeを重ねる |

6.693版には、新しいbranch-overlap model、追加学習済みmodel、追加Datasetは存在しない。「branch-overlap」は、既存routeを通ったwellにも最終midpoint hedgeを重ねる挙動を指している。

## 実測判断

```text
6.693 - 6.685 = +0.008
```

右側は手元再現で0.008悪化した。差は小さいが改善根拠がなく、広告値6.594も再現しなかったため、次を決定する。

- 6.685版を凍結controlとする。
- midpoint hedgeのstrength/cap/skip条件をLBで探索しない。
- 6.594という名称から新しいmodelがあると推測しない。
- 次はStage 16Bのtest-like branch-overlap validationへ進む。

## 再現方法

```bash
uv run rogii-frontier-diff \
  --config configs/experiment/stage16a_frontier_diff.yaml \
  --repo-dir . \
  --output-dir artifacts/stage16a_frontier_diff
```

生成物:

```text
frontier_diff.json
dependency_manifest.json
sanitation_audit.json
cell_manifest.json
notebook_diff.md
```
