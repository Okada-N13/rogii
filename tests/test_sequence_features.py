from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.models.sequence_features import (
    SEQUENCE_FEATURE_NAMES,
    build_sequence_well,
    feature_standardizer,
)


def _inputs(hidden_truth_shift: float = 0.0):
    n = 120
    md = np.arange(n, dtype=float)
    z = 1000.0 + 0.1 * md
    tvt = 120.0 - 0.02 * md
    tvt[60:] += hidden_truth_shift
    tvt_input = tvt.copy()
    tvt_input[60:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "Z": z,
            "X": 500.0 + md,
            "Y": 700.0 + 0.5 * md,
            "GR": 80.0 + np.sin(md / 8.0),
            "TVT": tvt,
            "TVT_input": tvt_input,
        }
    )
    type_tvt = np.linspace(50.0, 180.0, 300)
    typewell = pd.DataFrame({"TVT": type_tvt, "GR": 80.0 + np.sin(type_tvt / 8.0)})
    rows = np.arange(60, n)
    base = 120.0 - 0.018 * rows
    predictions = pd.DataFrame(
        {
            "id": [f"abcd_{row}" for row in rows],
            "well_id": "abcd",
            "row_index": rows,
            "fold": 0,
            "y_true": tvt[rows],
            "base_y_pred": base,
            "physics_y_pred": base + 1.0,
            "pred_conservative": base + 0.1,
        }
    )
    return horizontal, typewell, predictions


def test_sequence_features_do_not_use_hidden_tvt() -> None:
    horizontal, typewell, predictions = _inputs()
    original = build_sequence_well(horizontal, typewell, predictions)
    changed_horizontal, _, changed_predictions = _inputs(hidden_truth_shift=500.0)
    changed = build_sequence_well(changed_horizontal, typewell, changed_predictions)
    np.testing.assert_allclose(original.features, changed.features)
    assert not np.allclose(original.residual_target, changed.residual_target)
    assert original.features.shape[1] == len(SEQUENCE_FEATURE_NAMES)


def test_sequence_feature_standardizer_is_finite() -> None:
    horizontal, typewell, predictions = _inputs()
    well = build_sequence_well(horizontal, typewell, predictions)
    mean, scale = feature_standardizer([well])
    assert np.isfinite(mean).all()
    assert np.isfinite(scale).all()
    assert (scale > 0).all()
