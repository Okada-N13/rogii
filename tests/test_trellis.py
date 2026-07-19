from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.models.trellis import predict_trellis_correction


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    md = np.arange(100, dtype=float)
    z = -1000.0 - 0.9 * md
    tvt = 1100.0 + 0.92 * md
    gr = 100.0 + 25.0 * np.sin(tvt / 5.0)
    tvt_input = tvt.copy()
    tvt_input[50:] = np.nan
    horizontal = pd.DataFrame(
        {
            "well_id": "00abc123",
            "row_index": np.arange(len(md)),
            "MD": md,
            "Z": z,
            "TVT": tvt,
            "GR": gr,
            "TVT_input": tvt_input,
        }
    )
    typewell_tvt = np.arange(1080.0, 1210.0, 0.25)
    typewell = pd.DataFrame(
        {"TVT": typewell_tvt, "GR": 100.0 + 25.0 * np.sin(typewell_tvt / 5.0)}
    )
    target = horizontal.iloc[50:]
    prediction = pd.DataFrame(
        {
            "id": "00abc123_" + target["row_index"].astype(str),
            "well_id": "00abc123",
            "row_index": target["row_index"].to_numpy(),
            "MD": target["MD"].to_numpy(),
            "y_pred": target["TVT"].to_numpy() + 3.0,
            "y_true": target["TVT"].to_numpy(),
        }
    )
    return horizontal, typewell, prediction


def test_trellis_is_deterministic_and_does_not_read_hidden_tvt() -> None:
    horizontal, typewell, prediction = _inputs()
    config = {
        "tracking_stride": 2,
        "maximum_offset": 10.0,
        "offset_step": 1.0,
        "evidence_window": 5,
    }
    original, diagnostics = predict_trellis_correction(
        horizontal, typewell, prediction, config
    )
    modified_horizontal = horizontal.copy()
    modified_horizontal.loc[50:, "TVT"] += 999.0
    modified_prediction = prediction.copy()
    modified_prediction["y_true"] += 999.0
    changed, changed_diagnostics = predict_trellis_correction(
        modified_horizontal, typewell, modified_prediction, config
    )
    assert np.isfinite(original).all()
    assert np.array_equal(original, changed)
    assert diagnostics == changed_diagnostics
