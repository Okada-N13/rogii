from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.models.spatial import make_spatial_blocks, spatial_knn_predictions


def _wells() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": [f"well_{index}" for index in range(8)],
            "x": [0.0, 10.0, 20.0, 30.0, 1000.0, 1010.0, 1020.0, 1030.0],
            "y": [0.0, 5.0, -5.0, 3.0, 0.0, 5.0, -5.0, 3.0],
            "residual_bias": np.arange(8, dtype=float),
            "residual_slope_per_kft": np.arange(8, dtype=float) / 10.0,
            "n_rows": 100,
            "fold": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


def test_spatial_validation_predictions_do_not_use_validation_targets() -> None:
    wells = _wells()
    config = {"k_neighbors": 2, "distance_shrink_ft": 1000.0}
    original = spatial_knn_predictions(wells, "fold", config, seed=42).set_index("well_id")
    modified = wells.copy()
    modified.loc[modified["fold"] == 0, ["residual_bias", "residual_slope_per_kft"]] += 999.0
    changed = spatial_knn_predictions(modified, "fold", config, seed=42).set_index("well_id")
    validation = wells.loc[wells["fold"] == 0, "well_id"]
    columns = ["spatial_bias_pred", "spatial_slope_pred_per_kft"]
    assert np.allclose(original.loc[validation, columns], changed.loc[validation, columns])


def test_spatial_blocks_are_deterministic() -> None:
    wells = _wells()
    first = make_spatial_blocks(wells, n_blocks=2, seed=42)
    second = make_spatial_blocks(wells, n_blocks=2, seed=42)
    assert np.array_equal(first, second)
    assert first.nunique() == 2
