from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _frames(rows: int = 180) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = np.arange(rows)
    horizontal = pd.DataFrame(
        {
            "MD": index * 10.0,
            "Z": 1000.0 - index,
            "GR": 70.0 + 8.0 * np.sin(index / 9.0),
            "TVT": 5000.0 + index * 0.4,
        }
    )
    type_index = np.arange(600)
    typewell = pd.DataFrame(
        {
            "TVT": 4900.0 + type_index * 0.5,
            "GR": 70.0 + 8.0 * np.sin(type_index / 18.0),
        }
    )
    return horizontal, typewell


def test_strong_base_sequence_is_hidden_target_invariant() -> None:
    pytest.importorskip("torch")
    from rogii.cli.strong_base_emission import build_strong_base_sequence

    horizontal, typewell = _frames()
    cut = 80
    record = type(
        "Record",
        (),
        {
            "cut_id": "w__cut80",
            "well_id": "w",
            "cut_index": cut,
            "requested_fraction": 0.3,
            "stage16_fold": 1,
            "spatial_fold": 2,
            "typewell_fold": 3,
        },
    )()
    public = horizontal["TVT"].to_numpy(float)[cut:] - 1.0
    candidate_config = {
        "gr_sigma_multipliers": [1.3],
        "polynomial_degrees": [],
        "selector": {"particles": 3, "seeds": 1, "maximum_tracking_steps": 8},
        "top_pf_proxy": {
            "ridge_weight": 0.3,
            "selector_weight": 0.7,
            "projection_degree": 2,
            "projection_blend_weight": 0.75,
            "final_sp45_weight": 0.6,
        },
    }
    state_config = {
        "base_candidate": "top_pf_a130",
        "offset_min_ft": -10,
        "offset_max_ft": 10,
        "offset_step_ft": 1,
        "alignment_stride": 2,
        "windows": [5, 13, 25],
        "mix_windows": [13, 25],
        "mix_weights": [0.4, 0.6],
    }
    first = build_strong_base_sequence(
        record, horizontal, typewell, public, candidate_config, state_config
    )
    changed = horizontal.copy()
    changed.loc[cut:, "TVT"] += 9999.0
    second = build_strong_base_sequence(
        record, changed, typewell, public, candidate_config, state_config
    )
    np.testing.assert_array_equal(first.costs, second.costs)
    np.testing.assert_array_equal(first.row_features, second.row_features)
    np.testing.assert_array_equal(first.surface_y_pred, second.surface_y_pred)
    assert first.costs.shape[1] == 7


def test_rank_values_uses_explicit_offset_grid() -> None:
    pytest.importorskip("torch")
    from rogii.cli.strong_base_emission import (
        _improved_groups,
        _rank_values,
        build_strong_base_sequence,
    )

    horizontal, typewell = _frames()
    cut = 80
    record = type(
        "Record",
        (),
        {
            "cut_id": "w__cut80",
            "well_id": "w",
            "cut_index": cut,
            "requested_fraction": 0.3,
            "stage16_fold": 1,
            "spatial_fold": 2,
            "typewell_fold": 3,
        },
    )()
    public = horizontal["TVT"].to_numpy(float)[cut:] - 1.0
    candidates = {
        "gr_sigma_multipliers": [1.3],
        "polynomial_degrees": [],
        "selector": {"particles": 3, "seeds": 1, "maximum_tracking_steps": 8},
        "top_pf_proxy": {
            "ridge_weight": 0.3,
            "selector_weight": 0.7,
            "projection_degree": 2,
            "projection_blend_weight": 0.75,
            "final_sp45_weight": 0.6,
        },
    }
    state = {
        "base_candidate": "top_pf_a130",
        "offset_min_ft": -10,
        "offset_max_ft": 10,
        "offset_step_ft": 1,
        "alignment_stride": 2,
        "windows": [5, 13, 25],
        "mix_windows": [13, 25],
        "mix_weights": [0.4, 0.6],
    }
    sequence = build_strong_base_sequence(
        record, horizontal, typewell, public, candidates, state
    )
    offsets = np.arange(-10, 11, dtype=np.float32)
    logits = np.zeros((len(sequence.target_state), len(offsets)), np.float32)
    frame = _rank_values(sequence, logits, offsets, 0.25)
    assert np.allclose(frame["expected_offset"], 0.0)
    frame["branch_group_fold"] = 4
    count, report = _improved_groups(frame, "branch_group_fold")
    assert count in {0, 1} and "4" in report


def test_stage23b_notebook_is_clean_and_compiles() -> None:
    path = ROOT / "notebooks" / "580_run_stage23b_learned_emission.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "'-m','rogii.cli.strong_base_emission'" in text
    assert "'--device','cuda'" in text
    assert "PYTHONPATH" in text and "_driver.log" in text
    assert payload["metadata"]["stage23b"]["submission"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")
