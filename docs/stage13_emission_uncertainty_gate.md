# Stage 13: emission uncertainty gate

Stage 12Cでは最弱のpath profileでも通常`+0.0046`、spatial`+0.0528`、typewell`+0.0384`悪化し、制約を強くするほどRMSEとP90が単調に悪化した。したがってpath weightの追加探索は終了し、Stage 12Bのrow-wise expected offsetを基準とする。

Stage 13は再学習を行わない。各family自身のexpected correctionについて、補正絶対値、隣接行roughness、well平均補正量をtarget-free riskへ変換する。固定profileは弱い全体縮小、30/40 ft cap、上位10/20% riskだけの縮小、および両者の組み合わせである。

通常well OOFでは追加診断として、standard/spatial/typewell予測の5〜10% blendと、3モデル不一致・Stage 12B entropyによる縮小も評価する。ただし、standard予測を使った候補は厳密なspatial promotionには使わない。spatialとtypewellでは各family自身の予測とsurfaceだけから作ったlocal profileをnested選択する。

promotionには通常nested gain、bootstrap、P90、spatial/typewell nonworse、および全familyで安全な同一local profileを要求する。不合格なら決定論的なcap/gateを終了し、cross-fitted learned residual/uncertainty modelへ進む。
