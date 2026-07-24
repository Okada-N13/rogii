from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.cli.strong_base_ncc import _signal_counts, strong_base_costs


ROOT = Path(__file__).resolve().parents[1]


def test_strong_base_costs_are_target_invariant_and_finite() -> None:
    rows, cut = 140, 40
    index = np.arange(rows)
    horizontal = pd.DataFrame(
        {
            "MD": index * 10.0,
            "Z": 1000.0 - index,
            "GR": 70.0 + 8.0 * np.sin(index / 9.0),
            "TVT": 5000.0 + index * 0.4,
        }
    )
    horizontal.loc[60:65, "GR"] = np.nan
    type_index = np.arange(500)
    typewell = pd.DataFrame(
        {
            "TVT": 4950.0 + type_index * 0.5,
            "GR": 70.0 + 8.0 * np.sin(type_index / 18.0),
        }
    )
    base = horizontal["TVT"].to_numpy(float)[cut:] - 2.0
    config = {
        "offset_min_ft": -10,
        "offset_max_ft": 10,
        "offset_step_ft": 1,
        "alignment_stride": 2,
        "windows": [5, 13, 25],
        "mix_windows": [13, 25],
        "mix_weights": [0.4, 0.6],
    }
    first = strong_base_costs(horizontal, typewell, cut, base, config)
    changed = horizontal.copy()
    changed.loc[cut:, "TVT"] += 9999.0
    second = strong_base_costs(changed, typewell, cut, base, config)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    for name in first[3]:
        np.testing.assert_array_equal(first[3][name], second[3][name])
        assert np.isfinite(first[3][name]).all()


def test_signal_counts_handles_groups_without_valid_rows() -> None:
    frame = pd.DataFrame(
        {"fold": [0, 0, 1], "emission_valid": [False, False, True], "top10": [False, False, True]}
    )
    count, report = _signal_counts(frame, "fold", 0.2)
    assert count == 1
    assert report == {"0": 0.0, "1": 1.0}


def test_stage23a_notebook_is_clean_and_compiles() -> None:
    path = ROOT / "notebooks" / "570_run_stage23a_strong_base_ncc.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "rogii-strong-base-ncc" in text
    assert "stage21b_prefix_confidence_full_v001" in text
    assert payload["metadata"]["stage23a"]["submission"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")
