from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.cli.donor_ranker import FEATURE_COLUMNS, _feature_record, _training_mask


def test_training_mask_excludes_evaluation_fold_in_both_roles() -> None:
    rows = pd.DataFrame({"target_branch_fold": [0, 1, 1, 2], "donor_branch_fold": [1, 0, 2, 2]})
    mask = _training_mask(rows, 0)
    assert mask.tolist() == [False, False, True, True]


def test_donor_features_ignore_hidden_target_suffix() -> None:
    edge = pd.Series({
        "donor_rank": 1, "minimum_xyz_distance_ft": 10.0, "median_xyz_distance_ft": 20.0,
        "matched_prefix_points": 8, "mean_abs_gr_difference": 2.0, "typewell_gr_mean_difference": 3.0,
    })
    target_u = np.linspace(100.0, 110.0, 20)
    donor_u = np.linspace(99.0, 112.0, 20)
    distance = np.linspace(5.0, 25.0, 20)
    gr_delta = np.linspace(1.0, 4.0, 20)
    base = np.linspace(90.0, 95.0, 10)
    z = np.linspace(10.0, 12.0, 20)
    first, _ = _feature_record(edge, 0.5, 10, 10, target_u, donor_u, distance, gr_delta, base, z, 8)
    changed = target_u.copy()
    changed[10:] += 999.0
    second, _ = _feature_record(edge, 0.5, 10, 10, changed, donor_u, distance, gr_delta, base, z, 8)
    assert list(first) == FEATURE_COLUMNS
    assert first == second
