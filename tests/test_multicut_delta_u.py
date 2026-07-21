from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.data.multicut import (
    TARGET_COLUMNS,
    build_cut_record,
    build_multicut_records,
    feature_columns,
    make_cut_indices,
)
from rogii.models.delta_u_surface import build_row_predictions, crossfit_delta_u_coefficients


def _horizontal(well_id: str = "well0001", n_rows: int = 240, offset: float = 0.0) -> pd.DataFrame:
    row = np.arange(n_rows)
    md = row.astype(float) * 10.0
    z = 1000.0 - 0.18 * md + 2.0 * np.sin(row / 31.0)
    u = 12000.0 + offset + 0.012 * md + 0.0000015 * np.square(md)
    tvt = u - z
    return pd.DataFrame(
        {
            "well_id": well_id,
            "row_index": row,
            "MD": md,
            "X": offset * 100.0 + 0.7 * md,
            "Y": offset * 50.0 + 0.2 * md,
            "Z": z,
            "GR": 80.0 + offset + 12.0 * np.sin(row / 9.0),
            "TVT": tvt,
            "TVT_input": np.where(row < n_rows // 3, tvt, np.nan),
        }
    )


def _typewell(offset: float = 0.0) -> pd.DataFrame:
    tvt = np.linspace(10800.0, 12400.0, 300)
    return pd.DataFrame({"TVT": tvt, "GR": 75.0 + offset + 10.0 * np.sin(tvt / 35.0)})


def test_cut_indices_are_deterministic_and_respect_bounds() -> None:
    assert make_cut_indices(100, [0.1, 0.5, 0.5, 0.9], 20, 25) == [20, 50, 75]
    assert make_cut_indices(30, [0.5], 20, 20) == []


def test_cut_features_do_not_read_hidden_tvt() -> None:
    horizontal = _horizontal()
    original = build_cut_record(horizontal, _typewell(), 120)
    perturbed = horizontal.copy()
    perturbed.loc[120:, "TVT"] += 999.0
    changed = build_cut_record(perturbed, _typewell(), 120)
    columns = feature_columns(pd.DataFrame([original]))
    np.testing.assert_array_equal(
        np.asarray([original[column] for column in columns]),
        np.asarray([changed[column] for column in columns]),
    )
    assert any(not np.isclose(original[column], changed[column]) for column in TARGET_COLUMNS)


def test_delta_u_crossfit_is_well_isolated_and_builds_rows() -> None:
    frames: list[pd.DataFrame] = []
    horizontal_by_well: dict[str, pd.DataFrame] = {}
    for index in range(12):
        well_id = f"well{index:04d}"
        horizontal = _horizontal(well_id, offset=float(index))
        horizontal_by_well[well_id] = horizontal
        records = build_multicut_records(
            horizontal,
            _typewell(float(index)),
            {"fractions": [0.4, 0.65], "min_prefix_rows": 20, "min_suffix_rows": 20},
        )
        records["fold"] = index % 3
        frames.append(records)
    all_records = pd.concat(frames, ignore_index=True)
    coefficients, features = crossfit_delta_u_coefficients(
        all_records,
        "fold",
        {
            "max_iter": 20,
            "min_samples_leaf": 4,
            "max_leaf_nodes": 7,
            "regional": {"k_neighbors": 3},
        },
        seed=42,
    )
    assert len(coefficients) == len(all_records)
    assert coefficients["cut_id"].is_unique
    assert "regional_slope_correction" in features
    assert np.isfinite(coefficients[[f"pred_{column}" for column in TARGET_COLUMNS]]).all().all()
    predictions = build_row_predictions(
        coefficients,
        horizontal_by_well,
        "fold",
        {"max_eval_rows_per_cut": 40, "correction_cap_ft": 20.0},
    )
    assert predictions["well_id"].nunique() == 12
    assert len(predictions) <= 12 * 2 * 40
    assert np.isfinite(predictions[["base_y_pred", "y_pred", "y_true"]]).all().all()
    assert predictions.groupby("well_id")["fold"].nunique().eq(1).all()
