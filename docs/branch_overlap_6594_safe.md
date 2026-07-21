# 6.594 branch-overlap frontier

The newly supplied `hahaha-nondet-agi.ipynb` reports a public score of 6.594. Its code is identical to the earlier public V599/A130 notebook except for the final PF seed-branch hedge.

| Setting | Public V599 source | New 6.594 source |
|---|---:|---:|
| Branch strength | 1.00 | 0.60 |
| Absolute cap | 3 ft | 2 ft |
| Skip wells changed by earlier routes | Yes | No |
| Reported branch result | 3 skipped | 1 applied, 2 rejected by separation |

The public-score improvement therefore does not come from another trained model. It comes from allowing one already visible-prefix-adjusted test well to receive a smaller second correction. The final prediction mean moved by about 0.608 ft between the supplied runs, so this is a concentrated test-set intervention rather than a broad model gain.

`notebooks/240_kaggle_branch_overlap_6594_safe.ipynb` applies exactly this three-parameter change to the sanitized V599 notebook that scored 6.685. All target-transfer and leaderboard-probe removals remain intact. The original 6.594 score is not automatically transferable to the sanitized build, so its score must be measured separately.

## Kaggle execution

Use the same Kaggle inputs as the successful sanitized V599 run, keep Internet disabled, select P100, and run every cell from a fresh session. Submit only `/kaggle/working/submission.csv`.

Before submission, confirm that the branch log says one well was applied and two were skipped for separation. Also retain `BRANCH_OVERLAP_6594_SAFE_AUDIT`, especially its SHA-256 hash. If a different number of wells is applied, the stochastic PF path did not reproduce the public run and the resulting leaderboard comparison is not controlled.

Current measured public scores:

- sanitized MHA250SEP2: 6.874;
- sanitized V599/A130 branch-conservative: 6.685;
- unsanitized public branch-overlap source: 6.594;
- sanitized branch-overlap build: not measured yet.
