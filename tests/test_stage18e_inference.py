from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rogii.inference.stage18_retrieval import (
    FEATURE_COLUMNS, PortableRanker, _assigned_fold, _feature_record, export_hist_gradient_boosting,
)
from rogii.cli.donor_ranker import _feature_record as _training_feature_record


def test_existing_test_well_uses_frozen_fold_assignment() -> None:
    assignments = pd.DataFrame({"well_id": ["same", "donor"], "branch_group_fold": [3, 1]}).set_index("well_id")
    edges = pd.DataFrame({
        "donor_well_id": ["donor"], "minimum_xyz_distance_ft": [1.0], "matched_prefix_points": [24],
    })
    assert _assigned_fold("same", edges, assignments, {"n_folds": 5}) == (3, "frozen_training_assignment")


def test_inference_features_ignore_hidden_target_tvt() -> None:
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
    first, _, _ = _feature_record(edge, 0.5, 10, target_u, donor_u, distance, gr_delta, base, z, 8)
    changed = target_u.copy()
    changed[10:] += 999.0
    second, _, _ = _feature_record(edge, 0.5, 10, changed, donor_u, distance, gr_delta, base, z, 8)
    training, _ = _training_feature_record(edge, 0.5, 10, 10, target_u, donor_u, distance, gr_delta, base, z, 8)
    assert list(first) == FEATURE_COLUMNS
    assert first == second
    assert first == training


def test_portable_ranker_matches_sklearn_with_missing_values() -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(120, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame.iloc[::11, 4] = np.nan
    target = rng.normal(size=len(frame))
    model = HistGradientBoostingRegressor(max_iter=8, max_leaf_nodes=7, min_samples_leaf=5, random_state=9).fit(frame, target)
    portable = PortableRanker(export_hist_gradient_boosting(model))
    assert np.allclose(portable.predict(frame), model.predict(frame), atol=1e-10)
