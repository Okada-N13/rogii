from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.models.public_physics import (
    PhysicsSpec,
    apply_physics_spec,
    nested_select_predictions,
    robust_poly_predict,
)


def test_robust_poly_predict_resists_one_outlier() -> None:
    x = np.arange(100, dtype=float)
    y = 10.0 + 0.2 * x
    y[50] += 100.0
    prediction = robust_poly_predict(x, y, np.array([110.0]), degree=1)
    assert abs(prediction[0] - 32.0) < 0.5


def test_nested_selector_uses_training_folds_only_and_improves() -> None:
    wells = np.repeat([f"w{index}" for index in range(10)], 10)
    folds = np.repeat(np.arange(5), 20)
    truth = np.linspace(100.0, 120.0, len(wells))
    anchor = np.repeat(np.linspace(99.0, 109.0, 10), 10)
    base_delta = truth - anchor
    base_prediction = anchor + 1.02 * base_delta
    base = pd.DataFrame(
        {
            "well_id": wells,
            "MD": np.tile(np.arange(10), 10) + 1.0,
            "y_true": truth,
            "y_pred": base_prediction,
        }
    )
    features = pd.DataFrame(
        {
            "anchor_tvt": anchor,
            "base_delta": base_prediction - anchor,
            "poly_u_deg1_tail160": truth - base_prediction,
        }
    )
    specs = [PhysicsSpec(1.0), PhysicsSpec(delta_scale=1.0 / 1.02)]
    prediction, selections, ranking = nested_select_predictions(
        base, features, folds, specs, minimum_selection_gain=0.0
    )
    assert np.sqrt(np.mean(np.square(prediction - truth))) < 1e-10
    assert all(row["selected_index"] == 1 for row in selections)
    assert ranking[0]["index"] == 1


def test_physics_correction_is_faded_and_capped() -> None:
    base = pd.DataFrame({"MD": [0.0, 1000.0], "y_pred": [10.0, 10.0]})
    features = pd.DataFrame(
        {"base_delta": [0.0, 0.0], "poly_u_deg1_tail160": [100.0, 100.0]}
    )
    spec = PhysicsSpec(1.0, "poly_u_deg1_tail160", 0.5, 4.0, 10.0)
    prediction = apply_physics_spec(base, features, spec)
    assert prediction[0] == 10.0
    assert abs(prediction[1] - 12.0) < 1e-8

