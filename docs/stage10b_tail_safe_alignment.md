# Stage 10B: tail-safe alignment gate

Stage 10A produced a real but uneven signal. The fixed `loose × 0.2` specification improved pooled RMSE by about 0.061 and improved every standard fold, while the nested candidate worsened the p90 and worst-10% concentration metrics. Stage 10B retains that signal and tests whether conservative correction caps and visible-prefix confidence gates remove the tail damage.

Fifteen fixed profiles are evaluated, including weights between 0.1 and 0.2, conservative caps, and the all-fold-positive balanced branch. They use only the alignment correction and diagnostics available at inference time: prefix GR-shape correlation, the alignment active flag, and correction magnitude. No hidden TVT target is used for gating.

Selection is nested independently under standard five-fold and spatial six-fold partitions. A final inference profile must improve pooled RMSE by at least 0.02 and avoid worsening every fold in both partitions. Final promotion additionally requires at least 0.05 RMSE improvement in both validations, a negative paired-bootstrap upper bound, and non-worsening p90 and worst-10% SSE share.

Run `notebooks/200_run_stage10b_tail_safe_alignment.ipynb` in a CPU Colab session. It reuses the Stage 10A artifacts on Google Drive and performs no Kaggle submission.
