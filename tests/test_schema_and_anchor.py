from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rogii.data.schema import SchemaError, validate_horizontal_well
from rogii.models.anchor import predict_flat_surface_anchor, predict_last_tvt_anchor


def make_well() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": ["abc"] * 5,
            "row_index": range(5),
            "MD": [100.0, 101.0, 102.0, 103.0, 104.0],
            "X": [1.0] * 5,
            "Y": [2.0] * 5,
            "Z": [-90.0, -91.0, -92.0, -93.0, -94.0],
            "GR": [80.0, 81.0, np.nan, 83.0, 84.0],
            "TVT": [110.0, 111.0, 112.0, 113.0, 114.0],
            "TVT_input": [110.0, 111.0, np.nan, np.nan, np.nan],
        }
    )


def test_validate_and_predict_flat_surface() -> None:
    frame = make_well()
    stats = validate_horizontal_well(frame, split="train")
    prediction = predict_flat_surface_anchor(frame)
    flat_tvt = predict_last_tvt_anchor(frame)

    assert stats.n_known == 2
    assert stats.n_target == 3
    assert stats.anchor_md == 101.0
    assert prediction["id"].tolist() == ["abc_2", "abc_3", "abc_4"]
    assert prediction["surface_anchor"].unique().tolist() == [20.0]
    assert prediction["y_pred"].tolist() == [112.0, 113.0, 114.0]
    assert flat_tvt["y_pred"].tolist() == [111.0, 111.0, 111.0]


def test_schema_rejects_non_contiguous_mask() -> None:
    frame = make_well()
    frame.loc[3, "TVT_input"] = frame.loc[3, "TVT"]
    with pytest.raises(SchemaError, match="after hidden suffix"):
        validate_horizontal_well(frame, split="train")
