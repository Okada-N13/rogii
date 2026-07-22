# Stage 14B: extended residual correction and absolute-tail gate

Stage 14は通常RMSEを0.465、spatialを0.520、typewellを0.542改善し、bootstrapも全familyで明確に負だった。唯一の失敗はworst-10% SSE shareだったが、通常の総SSEは約7.17%減り、worst-10%絶対SSEも概算約5.36%減っている。構成比の上昇だけでRMSEモデルを棄却しないため、Stage 14Bは絶対tail指標へ置き換える。

再学習は行わない。保存済みgeneric/stacked cross-fit residualへ、weight 0.50〜1.00、cap 8〜16 ftの9 profileを適用する。profileはinner foldsでnested選択され、通常・spatial・typewellの全familyで同じgeneric specが安全かも確認する。

tail gateはworst-tail absolute SSE、well RMSE CVaR、P90、最大well RMSEを使う。worst-tail shareは診断として残せるがpromotion条件にはしない。全条件を通過した場合、全データemission/residual ensembleと独自test inference packageを構築する。
