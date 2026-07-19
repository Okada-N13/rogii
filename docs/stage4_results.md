# Stage 4 uncertainty guard and trellis results

## Outcome

Stage 4 combines two deterministic post-processors:

1. for the 20% of wells with the highest mean PF seed uncertainty, cap the Stage 3 correction to ±1 ft;
2. add 10% of a dynamic-programming trellis correction derived from horizontal GR and the paired typewell.

On the local reference OOF this produced:

| Model | Pooled RMSE | Well median | Well P90 | Max well |
|---|---:|---:|---:|---:|
| Stage 3 local reference | 12.299725 | 7.6068 | 17.2828 | 65.2876 |
| Stage 4 promoted | **12.204505** | **7.3845** | **17.2061** | **63.3934** |

Stage 4 improved pooled RMSE by `0.095219`, improved 472 of 773 wells, and improved all five folds. The well-paired bootstrap mean RMSE delta was `-0.092259`, with a 95% interval of `[-0.149368, -0.034722]`.

| Fold | Stage 3 | Stage 4 | Delta |
|---:|---:|---:|---:|
| 0 | 14.273797 | 14.262739 | -0.011058 |
| 1 | 11.363522 | 11.278626 | -0.084896 |
| 2 | 10.175380 | 9.940503 | -0.234878 |
| 3 | 12.736559 | 12.597866 | -0.138693 |
| 4 | 12.468930 | 12.414951 | -0.053979 |

The worst-5% SSE share improved slightly from `0.438942` to `0.438717`; the worst-10% share rose slightly from `0.563568` to `0.564780`. The guard is therefore useful but not a complete solution to concentrated tail error.

## Trellis component

The trellis first calibrates typewell GR gain and offset using only the known horizontal-well prefix. On every fourth target row it scores TVT offsets from -40 to +40 ft using a centred nine-point robust GR mismatch. Viterbi dynamic programming selects a continuous correction path with limited per-step jumps, a transition cost, a weak zero-offset prior, and a stronger zero-offset initialization. The path is interpolated to every target row and blended at 10%.

Complete hidden-suffix GR and trajectory values are visible competition inputs. Horizontal hidden TVT, formation columns, neighbouring wells, and contact overrides are never used. A synthetic regression test changes hidden TVT by +999 and verifies identical trellis corrections.

## Parameter selection

The local whole-OOF trellis optimum was near weight 0.175--0.20. Leave-one-fold-out selection chose 0.175 for four held folds and 0.225 for one. The promoted weight is deliberately reduced to 0.10 because it preserved the strongest joint improvement in pooled RMSE, median, P90, and maximum-well error.

For the combined guard/path search, nested folds selected uncertainty quantiles 0.75--0.80, correction caps of 1 ft in four folds and 3 ft in one, and path weights 0.15--0.20. The promoted values 0.80, 1 ft, and 0.10 are the conservative tail-protection point rather than the minimum whole-OOF score.

The integrated Stage 4 pass took `48.05` seconds on the local CPU reference machine. The local Stage 3 reference was composed from separate PF runs and scored `12.299725`, whereas native Colab Stage 3 scored `12.339250`. Native Colab Stage 4 should therefore be recorded separately as the canonical reproduction result.

The first native Colab Stage 4 run scored `12.228593`, improving its exact Stage 3 input by `0.110657`. It improved 454 of 773 wells; the paired bootstrap interval was `[-0.157746, -0.037241]`. Median well RMSE improved from `7.6360` to `7.3626`, and P90 improved from `17.5013` to `17.3698`. This native result is the canonical Stage 4 reference.
