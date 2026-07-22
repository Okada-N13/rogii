from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.cli.selector_gate import FEATURES, _feature_frame


def test_gate_features_exclude_target_metrics() -> None:
    forbidden = {"baseline_rmse", "selector_rmse", "baseline_sse", "selector_sse", "gain_target"}
    assert forbidden.isdisjoint(FEATURES)


def test_feature_frame_builds_gain_only_as_label() -> None:
    cuts = pd.DataFrame({
        "cut_id": ["a"], "well_id": ["w"], "replay_eligible": [False],
        "requested_fraction": [0.2], "cut_index": [20], "suffix_rows": [80],
        "baseline_sse": [320.0], "selector_sse": [80.0], "stage16_fold": [0],
    })
    audit = pd.DataFrame({
        "cut_id": ["a"], "well_id": ["w"], "selector_code": [0], "scale": [5.0],
        "configured_beam_weight": [0.0], "hold_weight": [0.2], "tracking_steps": [40],
        "particles": [96], "seeds": [8], "gr_sigma": [20.0], "likelihood_spread": [3.0],
        "suffix_rows": [80],
    })
    frame = _feature_frame(cuts, audit)
    assert np.isclose(frame.loc[0, "prefix_fraction"], 0.2)
    assert np.isclose(frame.loc[0, "gain_target"], 1.0)
    assert set(FEATURES).issubset(frame.columns)
