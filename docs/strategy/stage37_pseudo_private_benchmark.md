# Stage 37: frozen pseudo-private benchmark

## 目的

Stage 11--35ではdesign validation上の小さな改善がKaggleへ移らないケースが続いた。
Stage 37はモデル探索より先に、今後の全モデルが共通利用する固定pseudo-private benchmarkを作る。

## Stage 37A: split manifest

既存の安全な分割規則を再利用し、新しい乱数抽選は行わない。Stage 24の分割artifactが
Driveに残っていない場合は、同じStage 17/21入力、seed 42、foldごとの
training 100 / confirmation 24という固定条件から決定論的に再構築する。

- training: Stage 24で固定した500 wellsに属する全primary replay-eligible cuts
- design validation: Stage 21Bで使用済みの62 cuts / 58 wells
- confirmation: 未使用120 wellsの固定cut。IDとcut位置だけをロック

三役のwell集合は完全に分離する。各cutへstandard、spatial、typewell、
branch-group foldを付与する。trainingは`0.18, 0.22, 0.26, 0.30, 0.34`を
すべて含まなければならない。

Stage 37Aはtrain CSVを開かず、TVT・予測・残差を一切計算しない。
したがってconfirmation targetを偶発的に参照しない。

## Stage 37B以降

Stage 37Aのmanifest hashを固定した後、trainingとdesignだけについてtop-PF相当baseを
materializeする。confirmationはモデル・profile・gateがdesign上の全条件を通過した時に
一度だけ開封する。

今後の提出条件:

- designでstandard 5/5
- spatial 5/6以上
- typewell 5/5
- branch group 5/5
- well bootstrap上限が0未満
- P90とworst-tailが非悪化
- 最後にconfirmationでも改善

Stage 37 manifest作成自体はCPUで短時間に完了し、submissionは作らない。
