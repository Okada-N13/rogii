from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.models.public_blend import (
    BlendSpec,
    align_package_ground_truth,
    nested_select_blend,
)


def test_package_ground_truth_is_reordered_and_verified() -> None:
    base = pd.DataFrame({"id": ["a", "b"], "y_true": [10.0, 20.0]})
    package = pd.DataFrame(
        {
            "id": ["b", "a"],
            "last_known_TVT": [15.0, 5.0],
            "target_tvt": [20.0, 10.0],
        }
    )
    order, report = align_package_ground_truth(base, package)
    assert order.tolist() == [1, 0]
    assert report["id_order_matches"] is False
    assert report["target_max_abs_difference"] == 0.0


def test_nested_public_blend_selects_decorrelated_branch() -> None:
    rng = np.random.default_rng(11)
    n = 500
    truth = np.linspace(100.0, 120.0, n)
    base_error = rng.normal(0.0, 2.0, n)
    branch_error = rng.normal(0.0, 1.0, n)
    base = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(n)],
            "well_id": np.repeat([f"w{i}" for i in range(50)], 10),
            "MD": np.tile(np.arange(10), 50),
            "y_true": truth,
            "y_pred": truth + base_error,
        }
    )
    branches = {"package": truth + branch_error}
    folds = np.repeat(np.arange(5), 100)
    specs = [BlendSpec("package", weight) for weight in (0.1, 0.3, 0.5)]
    prediction, selections, ranking = nested_select_blend(
        base, branches, folds, specs, minimum_selection_gain=0.01
    )
    base_rmse = np.sqrt(np.mean(np.square(base_error)))
    candidate_rmse = np.sqrt(np.mean(np.square(prediction - truth)))
    assert candidate_rmse < base_rmse
    assert all(row["selected_spec"] is not None for row in selections)
    assert ranking[0]["weight"] == 0.5


def test_robust_selector_rejects_candidate_that_hurts_one_inner_fold() -> None:
    truth = np.zeros(60)
    folds = np.repeat(np.arange(3), 20)
    base_prediction = np.ones(60)
    branch = np.zeros(60)
    branch[folds == 1] = 3.0
    base = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(60)],
            "well_id": np.repeat([f"w{i}" for i in range(6)], 10),
            "MD": np.tile(np.arange(10), 6),
            "y_true": truth,
            "y_pred": base_prediction,
        }
    )
    prediction, selections, _ = nested_select_blend(
        base,
        {"unstable": branch},
        folds,
        [BlendSpec("unstable", 1.0)],
        minimum_selection_gain=0.0,
        require_all_training_folds_improve=True,
    )
    # Folds whose selection data includes the harmed fold must keep the base.
    assert selections[0]["selected_spec"] is None
    assert selections[2]["selected_spec"] is None
    assert np.all(prediction[folds != 1] == 1.0)
