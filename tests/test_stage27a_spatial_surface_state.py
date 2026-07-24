from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.models.spatial_surface_state import anchored_spatial_plane, guarded_spatial_blend


ROOT = Path(__file__).resolve().parents[1]


def _horizontal() -> pd.DataFrame:
    md = np.arange(12, dtype=float) * 100.0
    x = md.copy()
    y = np.zeros_like(md)
    z = 0.1 * md
    u = 1000.0 + 0.02 * x
    return pd.DataFrame({"MD": md, "X": x, "Y": y, "Z": z, "TVT": u - z})


def test_spatial_plane_recovers_planar_surface_without_suffix_target() -> None:
    horizontal = _horizontal()
    prediction, gradient = anchored_spatial_plane(horizontal, 7, window_ft=1000.0, ridge=1e-8)
    assert np.allclose(prediction, horizontal["TVT"].to_numpy()[7:], atol=1e-5)
    changed = horizontal.copy()
    changed.loc[7:, "TVT"] += 999.0
    changed_prediction, changed_gradient = anchored_spatial_plane(
        changed, 7, window_ft=1000.0, ridge=1e-8
    )
    assert np.array_equal(prediction, changed_prediction)
    assert np.array_equal(gradient, changed_gradient)


def test_guarded_blend_respects_cap_and_anchor_ramp() -> None:
    base = np.zeros(5)
    plane = np.full(5, 100.0)
    prediction = guarded_spatial_blend(base, plane, weight=0.5, cap_ft=4.0, ramp_rows=2)
    assert np.all(prediction <= 4.0)
    assert 0.0 < prediction[0] < prediction[-1] < 4.0


def test_stage27a_notebook_is_clean_and_reserved_safe() -> None:
    path = ROOT / "notebooks" / "640_run_stage27a_spatial_surface_state.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "'-m','rogii.cli.spatial_surface_state'" in text
    assert "stage27a_spatial_surface_state.yaml" in text
    assert "PYTHONPATH" in text
    assert payload["metadata"]["stage27a"]["reserved_confirmation_used"] is False
    assert payload["metadata"]["stage27a"]["submission"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")

