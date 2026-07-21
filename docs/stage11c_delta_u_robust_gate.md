# Stage 11C: robust delta-U gate

Stage 11 improved the fixed pseudo-test benchmark from 21.7596 to 17.6718 at weight 0.35. Every ordinary fold improved by about four feet, spatial and typewell holdouts improved by more than three feet, and the paired-well bootstrap interval was entirely negative. The only failed gate was worst-10% SSE share.

That share is a concentration diagnostic rather than an absolute-tail loss. When total SSE falls quickly, the share may rise even if the worst wells improve in absolute terms. Stage 11C therefore retains the share as a report but gates on:

- absolute SSE of the worst 10% wells;
- mean RMSE of those wells (well-level CVaR);
- well RMSE P90 and maximum;
- pooled and per-fold RMSE.

## Nested selection

The candidate grid is fixed before Stage 11C runs:

- correction weights: 0.35, 0.50, 0.75, 1.00;
- raw correction caps: 30, 40, 50 ft.

For each outer fold, its rows are removed. A profile is eligible only if it improves the remaining pooled rows, does not worsen any remaining inner fold, and does not worsen absolute worst-tail SSE, CVaR, or P90. The lowest-RMSE eligible profile is then applied to the untouched outer fold. If none is eligible, that fold retains the prefix baseline.

This procedure is repeated independently for ordinary, six-block spatial, and typewell-signature folds. A single all-train inference profile is reported only if it improves every fold and all absolute-tail gates in all three families.

## Cut audit

The nested ordinary OOF is also split into the four pseudo-test cut fractions. All four must improve. This prevents a model that works only with a long known prefix from being promoted to the alignment stage.

## Colab execution

Open `notebooks/260_run_stage11c_delta_u_robust_gate.ipynb`. It is standalone and reuses `stage11_multicut_delta_u_full_v001` from Drive. If that artifact is missing, the notebook rebuilds Stage 11 once. Use a CPU/high-RAM runtime; GPU is unused.

Stage 11C writes:

```text
MyDrive/kaggle/rogii/artifacts/stage11c_delta_u_robust_gate_full_v001
```

No submission is generated. `promoted_to_stage12: True` authorizes the raw-NCC and learned-emission alignment benchmark only.
