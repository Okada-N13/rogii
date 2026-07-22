from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from rogii.models.raw_ncc import alignment_costs, benchmark_cut, rolling_ncc_cost
from rogii.models.emission_features import CANDIDATE_CHANNELS, build_emission_sequence, feature_invariance


def test_rolling_ncc_prefers_matching_sequence() -> None:
    observed = np.sin(np.arange(101) / 7.0)
    expected = np.column_stack([observed, observed[::-1]])
    cost = rolling_ncc_cost(observed, expected, np.ones_like(expected, dtype=bool), 13)
    assert float(np.median(cost[20:-20, 0])) < 0.05
    assert float(np.median(cost[20:-20, 0])) < float(np.median(cost[20:-20, 1]))


def _case() -> tuple[SimpleNamespace, pd.DataFrame, pd.DataFrame]:
    row = np.arange(220)
    md = row.astype(float)
    base_tvt = 1000.0 + md
    true_tvt = base_tvt.copy()
    true_tvt[60:] += 10.0
    type_tvt = np.linspace(900.0, 1300.0, 2000)
    type_gr = 80.0 + 15.0 * np.sin(type_tvt / 5.0) + 6.0 * np.sin(type_tvt / 13.0)
    horizontal = pd.DataFrame(
        {
            "well_id": "well0001",
            "row_index": row,
            "MD": md,
            "X": md,
            "Y": 0.0,
            "Z": 0.0,
            "GR": np.interp(true_tvt, type_tvt, type_gr),
            "TVT": true_tvt,
        }
    )
    record = SimpleNamespace(
        well_id="well0001",
        cut_id="well0001__cut60",
        cut_index=60,
        cut_fraction=60 / 220,
        anchor_md=59.0,
        anchor_u=1059.0,
        prefix_u_slope_per_kft=1000.0,
        pred_target_slope_correction=0.0,
        pred_target_curvature=0.0,
        fold=0,
    )
    return record, horizontal, pd.DataFrame({"TVT": type_tvt, "GR": type_gr})


def test_raw_ncc_recovers_a_known_constant_offset() -> None:
    record, horizontal, typewell = _case()
    config = {
        "offset_min_ft": -20.0,
        "offset_max_ft": 20.0,
        "offset_step_ft": 2.0,
        "alignment_stride": 1,
        "max_eval_rows_per_cut": 200,
        "windows": [5, 13, 25],
        "mix_windows": [13, 25],
        "mix_weights": [0.4, 0.6],
    }
    result = benchmark_cut(
        record,
        horizontal,
        typewell,
        config,
        weight=0.75,
        correction_cap_ft=50.0,
        fold_column="fold",
    )
    assert float(result["ncc_mix_offset"].median()) == 10.0
    assert float(result["ncc_mix_top5"].mean()) > 0.9
    assert float(result["ncc_mix_y_pred"].sub(result["y_true"]).abs().median()) < 0.1


def test_alignment_costs_ignore_hidden_tvt() -> None:
    record, horizontal, typewell = _case()
    config = {
        "offset_min_ft": -20.0,
        "offset_max_ft": 20.0,
        "offset_step_ft": 2.0,
        "alignment_stride": 2,
        "windows": [5, 13, 25],
        "mix_windows": [13, 25],
        "mix_weights": [0.4, 0.6],
    }
    original = alignment_costs(
        record, horizontal, typewell, config, weight=0.75, correction_cap_ft=50.0
    )
    changed = horizontal.copy()
    changed.loc[60:, "TVT"] += 999.0
    perturbed = alignment_costs(
        record, changed, typewell, config, weight=0.75, correction_cap_ft=50.0
    )
    np.testing.assert_array_equal(original[1], perturbed[1])
    for name in original[3]:
        np.testing.assert_array_equal(original[3][name], perturbed[3][name])


def test_emission_sequence_has_paired_gr_channels_and_is_target_invariant() -> None:
    record, horizontal, typewell = _case()
    record.spatial_fold = 1
    record.typewell_fold = 2
    config = {
        "offset_min_ft": -20.0,
        "offset_max_ft": 20.0,
        "offset_step_ft": 2.0,
        "alignment_stride": 2,
        "max_rows_per_cut": 32,
        "windows": [5, 13, 25],
        "mix_windows": [13, 25],
        "mix_weights": [0.4, 0.6],
    }
    sequence = build_emission_sequence(
        record, horizontal, typewell, config, weight=0.75, correction_cap_ft=50.0
    )
    assert sequence.costs.shape == (32, len(CANDIDATE_CHANNELS), 21)
    assert sequence.row_features.shape == (32, 4)
    assert sequence.valid.any()
    assert feature_invariance(
        record, horizontal, typewell, config, weight=0.75, correction_cap_ft=50.0
    )
