from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.models.emission_uncertainty import apply_uncertainty_profile, uncertainty_features


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cut_id": ["a"] * 4 + ["b"] * 4,
            "well_id": ["w1"] * 4 + ["w2"] * 4,
            "row_index": list(range(4)) * 2,
            "surface_y_pred": np.zeros(8),
            "y_pred": [1.0, 2.0, 3.0, 20.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def test_uncertainty_features_rank_large_rough_correction_as_risky() -> None:
    frame = _frame()
    features = uncertainty_features(frame)
    assert int(features["local_risk"].argmax()) == 3
    assert features.loc[3, "correction_roughness"] == 17.0


def test_uncertainty_profile_caps_and_shrinks_only_high_risk_rows() -> None:
    frame = _frame()
    features = uncertainty_features(frame)
    prediction = apply_uncertainty_profile(
        frame,
        features,
        {"correction_cap": 10.0, "risk_threshold": 0.8, "high_risk_scale": 0.5},
    )
    assert prediction[3] == 5.0
    assert prediction[-1] == 1.0
