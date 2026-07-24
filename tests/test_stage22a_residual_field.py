from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.cli.prefix_residual_field import _correction_prediction, residual_features


ROOT = Path(__file__).resolve().parents[1]


def _horizontal(rows: int = 100) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "MD": index * 10.0,
            "Z": 1000.0 - index,
            "GR": 70.0 + 5.0 * np.sin(index / 8.0),
            "TVT": 5000.0 + index * 0.4,
        }
    )


def test_residual_features_are_hidden_target_invariant() -> None:
    horizontal = _horizontal()
    cut = 40
    suffix = len(horizontal) - cut
    base = np.linspace(5016.0, 5040.0, suffix)
    candidates = {
        "top_pf_a130": base,
        "top_pf_a100": base - 1.0,
        "top_pf_a160": base + 2.0,
        "selector_a130": base + np.sin(np.arange(suffix) / 10.0),
        "public_oof": base - 0.5,
    }
    first, names = residual_features(horizontal, cut, candidates, "top_pf_a130", 0.3)
    changed = horizontal.copy()
    changed.loc[cut:, "TVT"] += 9999.0
    second, changed_names = residual_features(changed, cut, candidates, "top_pf_a130", 0.3)
    np.testing.assert_array_equal(first, second)
    assert names == changed_names
    assert first.shape == (suffix, len(names))


def test_correction_is_ramped_capped_and_finite() -> None:
    base = np.full(20, 100.0)
    prediction = _correction_prediction(base, np.full(20, 20.0), 0.25, 8.0, 5.0)
    assert np.isfinite(prediction).all()
    assert np.all(prediction >= base)
    assert np.max(prediction - base) <= 8.0
    assert prediction[0] - base[0] < prediction[-1] - base[-1]


def test_stage22a_notebook_is_clean_and_compiles() -> None:
    path = ROOT / "notebooks" / "560_run_stage22a_residual_field.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "rogii-prefix-residual-field" in text
    assert "stage21a_prefix_router_full_v001" in text
    assert "stage21b_prefix_confidence_full_v001" in text
    assert payload["metadata"]["stage22a"]["submission"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")
