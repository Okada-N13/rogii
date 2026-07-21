from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.models.multiscale_alignment import predict_multiscale_alignment


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    typewell_tvt = np.arange(900.0, 1300.0, 0.25)
    typewell_gr = (
        100.0
        + 22.0 * np.sin(typewell_tvt / 7.3)
        + 11.0 * np.sin(typewell_tvt / 2.7)
        + 5.0 * np.cos(typewell_tvt / 17.0)
    )
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})
    rows = np.arange(180)
    truth = 980.0 + 0.75 * rows + 0.002 * rows**2
    gr = np.interp(truth, typewell_tvt, typewell_gr)
    tvt_input = truth.copy()
    tvt_input[70:] = np.nan
    horizontal = pd.DataFrame({
        "row_index": rows, "MD": rows.astype(float), "TVT": truth,
        "TVT_input": tvt_input, "GR": gr,
    })
    hidden = rows[70:]
    prediction = pd.DataFrame({
        "row_index": hidden,
        "base_y_pred": truth[hidden] - 5.0,
    })
    return horizontal, typewell, prediction


def test_multiscale_alignment_recovers_shift_without_hidden_target() -> None:
    horizontal, typewell, prediction = _inputs()
    config = {
        "tracking_stride": 2, "maximum_offset": 10.0, "offset_step": 1.0,
        "half_windows": [2, 5, 10], "scale_weights": [0.2, 0.4, 0.4],
        "amplitude_weight": 0.1, "max_jump_states": 2,
        "transition_penalty": 0.01, "zero_penalty": 0.0, "initial_penalty": 0.0,
        "minimum_cost_gain": -1.0, "minimum_prefix_correlation": -1.0,
    }
    correction, diagnostics = predict_multiscale_alignment(horizontal, typewell, prediction, config)
    modified = horizontal.copy()
    modified.loc[70:, "TVT"] += 9999.0
    changed, changed_diagnostics = predict_multiscale_alignment(modified, typewell, prediction, config)
    assert np.median(correction[20:-20]) >= 4.0
    assert np.array_equal(correction, changed)
    assert diagnostics == changed_diagnostics
    assert diagnostics["active"] == 1.0
