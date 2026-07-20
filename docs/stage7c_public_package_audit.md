# Stage 7C: 公開model package OOF監査

ravaghi branchへの残差HGBと低自由度physics補正はいずれもnested OOFで不採用となった。次はbranch単体を補正せず、誤差の異なる公開model packageをblendできるか調べる。

対象は以下の2 Datasetである。

- `fleongg/rogii-claude-models-pub`
- `pilkwang/rogii-model-package`

fleongg packageには196特徴の3 LightGBMがあるが、公開情報だけでは各pickleがfull-data modelかfold modelか、OOFが保存されているかを断定できない。pilkwang packageにはmanifest上で全3,783,989行のOOFがあるため、実ファイルと行整列を確認する。

[120_colab_public_package_audit.ipynb](../../notebooks/120_colab_public_package_audit.ipynb)は次を自動記録する。

- 全ファイル名とsize
- OOF/fold/prediction/blend/manifest関連ファイル
- CSV/Parquetの列、行数、sample
- NPY/NPZのshapeとdtype
- ID列がある場合の順序SHA-256とravaghi baseとの一致
- JSON manifest/config/metricsの要約

行数が一致するだけではblendを許可しない。ID順序一致、またはmanifestとtarget照合による同一行順序の証明が必要である。学習内predictionしかないbranchもblend OOFには使用しない。

この監査はモデルを学習せず、Kaggle提出も生成しない。監査結果から正規OOF列が特定できた後、nested nonnegative blend gateを別runとして実装する。

