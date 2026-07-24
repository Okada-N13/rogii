from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.cli.expanded_residual_field import (
    expanded_residual_features,
    smooth_residual_target,
)


ROOT = Path(__file__).resolve().parents[1]


def _horizontal() -> pd.DataFrame:
    md = np.arange(16, dtype=float) * 100.0
    return pd.DataFrame(
        {
            "MD": md, "X": md, "Y": 0.1 * md, "Z": 0.05 * md,
            "GR": 80.0 + np.sin(md / 200.0),
            "TVT": 1000.0 - 0.03 * md,
        }
    )


def test_expanded_features_ignore_suffix_target() -> None:
    horizontal = _horizontal()
    cut = 8
    suffix = len(horizontal) - cut
    candidates = {
        "top_pf_a130": np.linspace(970.0, 940.0, suffix),
        "top_pf_a100": np.linspace(971.0, 942.0, suffix),
    }
    first, names = expanded_residual_features(
        horizontal, cut, candidates, "top_pf_a130", 0.3,
        plane_window_ft=1200.0, plane_ridge=0.05,
    )
    changed = horizontal.copy()
    changed.loc[cut:, "TVT"] += 999.0
    second, changed_names = expanded_residual_features(
        changed, cut, candidates, "top_pf_a130", 0.3,
        plane_window_ft=1200.0, plane_ridge=0.05,
    )
    assert names == changed_names
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_smoothed_target_caps_outliers() -> None:
    target = smooth_residual_target(np.array([0.0, 100.0, 0.0]), 3, 10.0)
    assert np.all(np.abs(target) <= 10.0)
    assert np.all(target > 0.0)


def test_stage28a_notebook_is_clean_and_reserve_safe() -> None:
    path = ROOT / "notebooks" / "650_run_stage28a_expanded_residual_field.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "rogii-expanded-residual-field" in text
    assert "stage28a_expanded_residual_field.yaml" in text
    assert "stage24a_scaled_emission_manifest_v003" in text
    assert payload["metadata"]["stage28a"]["reserved_confirmation_used"] is False
    assert payload["metadata"]["stage28a"]["submission"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")

