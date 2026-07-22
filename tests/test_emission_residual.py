from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.models.emission_residual import GENERIC_FEATURES, STACKED_FEATURES, generic_residual_features, stacked_residual_features, target_invariance_check


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cut_id": ["a"] * 4 + ["b"] * 4,
            "well_id": ["w1"] * 4 + ["w2"] * 4,
            "MD": [0.0, 10.0, 20.0, 30.0] * 2,
            "cut_fraction": [0.5] * 8,
            "surface_y_pred": np.arange(8, dtype=float),
            "y_pred": np.arange(8, dtype=float) + np.array([0, 1, 2, 3, 1, 1, 1, 1]),
            "y_true": np.arange(8, dtype=float) + 2.0,
        }
    )


def test_generic_residual_features_are_target_invariant_and_finite() -> None:
    frame = _frame()
    features = generic_residual_features(frame)
    assert list(features.columns) == list(GENERIC_FEATURES)
    assert np.isfinite(features.to_numpy()).all()
    assert target_invariance_check(frame)


def test_stacked_features_include_target_free_branch_disagreement() -> None:
    frame = _frame()
    spatial = frame["y_pred"].to_numpy(float) + 2.0
    typewell = frame["y_pred"].to_numpy(float) - 1.0
    features = stacked_residual_features(frame, spatial, typewell, np.linspace(1.0, 2.0, len(frame)))
    assert list(features.columns) == list(STACKED_FEATURES)
    np.testing.assert_allclose(features["spatial_minus_standard"], 2.0)
    np.testing.assert_allclose(features["typewell_minus_standard"], -1.0)
