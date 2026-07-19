from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.models.registry import predict_model
from rogii.models.trend import predict_guarded_trend


def make_linear_well() -> pd.DataFrame:
    md = np.arange(30, dtype=float)
    tvt = 100.0 + 0.02 * md
    tvt_input = tvt.copy()
    tvt_input[20:] = np.nan
    return pd.DataFrame(
        {
            "well_id": ["linear"] * len(md),
            "row_index": np.arange(len(md)),
            "MD": md,
            "X": 1.0,
            "Y": 2.0,
            "Z": -50.0 - 0.1 * md,
            "GR": 80.0,
            "TVT": tvt,
            "TVT_input": tvt_input,
        }
    )


def test_linear_tvt_trend_recovers_slope() -> None:
    frame = make_linear_well()
    prediction = predict_guarded_trend(
        frame,
        {
            "target": "tvt",
            "degree": 1,
            "window_ft": 100,
            "minimum_points": 10,
            "ridge": 0.0,
            "slope_clip": 0.03,
        },
    )
    assert np.allclose(prediction["y_pred"], prediction["y_true"], atol=1e-8)
    assert np.isclose(prediction["trend_slope"].iloc[0], 0.02)


def test_quadratic_trend_respects_delta_guard() -> None:
    frame = make_linear_well()
    prediction = predict_guarded_trend(
        frame,
        {
            "target": "tvt",
            "degree": 2,
            "window_ft": 100,
            "minimum_points": 10,
            "ridge": 0.0,
            "slope_clip": 1.0,
            "curvature_clip": 1.0,
            "strength": 100.0,
            "max_delta_ft": 0.5,
        },
    )
    anchor = frame.loc[19, "TVT_input"]
    assert np.max(np.abs(prediction["y_pred"] - anchor)) <= 0.5 + 1e-12


def test_fixed_blend_matches_weighted_components() -> None:
    frame = make_linear_well()
    trend_config = {
        "name": "guarded_trend",
        "target": "tvt",
        "degree": 1,
        "window_ft": 100,
        "minimum_points": 10,
        "ridge": 0.0,
    }
    anchor = predict_model(frame, {"name": "last_tvt_anchor"})
    trend = predict_model(frame, trend_config)
    blend = predict_model(
        frame,
        {
            "name": "fixed_blend",
            "components": [
                {"weight": 0.75, "model": {"name": "last_tvt_anchor"}},
                {"weight": 0.25, "model": trend_config},
            ],
        },
    )
    expected = 0.75 * anchor["y_pred"].to_numpy() + 0.25 * trend["y_pred"].to_numpy()
    assert np.allclose(blend["y_pred"], expected)

