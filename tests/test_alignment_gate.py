from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.models.alignment_gate import apply_alignment_profile


def test_alignment_profile_caps_and_rejects_low_confidence_wells() -> None:
    frame = pd.DataFrame({
        "well_id": ["a", "a", "b", "b"],
        "base_y_pred": [100.0, 101.0, 200.0, 201.0],
        "correction_loose": [10.0, -10.0, 5.0, -5.0],
    })
    diagnostics = pd.DataFrame({
        "well_id": ["a", "b"], "branch": ["loose", "loose"],
        "active": [1.0, 1.0], "prefix_shape_corr": [0.4, 0.05],
        "cost_gain_per_step": [0.2, 0.2],
    })
    prediction, report = apply_alignment_profile(frame, diagnostics, {
        "branch": "loose", "weight": 0.2, "correction_cap": 6.0,
        "minimum_prefix_correlation": 0.1,
    })
    assert np.allclose(prediction, [101.2, 99.8, 200.0, 201.0])
    assert report["selected_wells"] == 1
    assert report["rejected_wells"] == 1
    assert np.isclose(report["max_abs_move"], 1.2)
