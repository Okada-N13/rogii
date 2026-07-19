# Stage 1 — guarded trend baselines

## Result

All results use the fixed 773-well hidden-suffix evaluation and the same five well folds as `baseline_anchor_full_v001`.

| Run | Pooled RMSE | Delta vs anchor | Interpretation |
|---|---:|---:|---|
| Last-known TVT anchor | 15.909853 | — | Reference |
| Robust linear TVT | 35.719096 | +19.809243 | Prefix slope does not extrapolate legally |
| Guarded quadratic TVT | 51.865541 | +35.955688 | Curvature remains unstable |
| Robust linear surface | 37.017687 | +21.107834 | Useful only as a weak decorrelated correction |
| Guarded quadratic surface | 115.701548 | +99.791695 | Standalone use is rejected |
| Fixed 95/4/1 blend | **15.700508** | **-0.209344** | Promoted as a weak baseline correction |

The promoted blend is:

```text
0.95 × last-known TVT anchor
+ 0.04 × robust linear surface trend
+ 0.01 × guarded quadratic surface trend
```

It improved all five folds:

| Fold | Anchor | Trend blend |
|---:|---:|---:|
| 0 | 17.870733 | 17.468689 |
| 1 | 14.674600 | 14.367166 |
| 2 | 15.640527 | 15.612290 |
| 3 | 17.127863 | 16.984718 |
| 4 | 13.850122 | 13.715257 |

Additional controls:

- median well RMSE: 10.665141 → 10.569538
- P90 well RMSE: 22.972537 → 22.914607
- worst 10% SSE share: 52.4763% → 51.8511%
- improved wells: 462 / 773
- paired well bootstrap mean delta: -0.1476 RMSE
- paired bootstrap 95% interval: [-0.2575, -0.0386]
- five-fold nested weight selection: pooled RMSE 15.742048

The fixed-weight 15.700508 value is a development OOF result because the coarse weights were selected using these OOF predictions. The nested value is the more conservative estimate of generalization. Both improve over the anchor.

## Decision

- Promote `configs/experiment/trend_blend.yaml` as the Stage 1 baseline.
- Keep the four standalone trend configs as diagnostic controls only.
- Do not increase trend weights: the standalone models have large tail failures.
- Use the trend blend as one weak component or prior when the Particle Filter is introduced.

