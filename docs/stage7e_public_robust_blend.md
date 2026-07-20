# Stage 7E: spatially robust public blend gate

Stage 7Dは通常OOFを`-0.3277`改善し、5/5 folds、bootstrap、P90、worst-wellを通過した。空間全体も`-0.3056`改善したが、必要な5/6 blocksの一貫性を満たさず、最終`promoted`はfalseだった。

Stage 7Dのouter selectionはpostprocessed packageとTCNを切り替えていた。Stage 7Eではこの自由度を削除し、全OOF inference specかつ通常・空間selectionの多数派だった`package_postprocessed`だけに固定する。

候補weightは`0.20 / 0.30 / 0.40`。各outer fold/blockの選択データに含まれるすべてのinner fold/blockでbaseを悪化させないweightだけをeligibleとし、その中でpooled RMSEが最良かつ`0.02`以上改善するものを選ぶ。該当候補がなければそのouter groupはbaseを維持する。

[140_colab_public_robust_blend_gate.ipynb](../../notebooks/140_colab_public_robust_blend_gate.ipynb)をColab CPUで実行する。

これはStage 7Dと同じOOF母集団を使った事後的なrobustness確認であり、独立した新データではない。通過してもSafe MHA全体へ40%を直接混ぜず、OOFで対応するravaghi branch内だけを置換する。Kaggle最終出力に対する実効package weightはその下流blend weightによって縮小される。

