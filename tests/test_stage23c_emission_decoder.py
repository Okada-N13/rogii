from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_posterior_features_are_finite_and_ordered() -> None:
    pytest.importorskip("torch")
    from rogii.cli.emission_decoder import posterior_features

    offsets = np.arange(-2, 3, dtype=np.float32)
    logits = np.array([[0, 0, 5, 0, 0], [5, 0, 0, 0, 0]], dtype=np.float32)
    affine, summary = posterior_features(logits, offsets)
    assert affine.shape == (2, 1)
    assert summary.shape == (2, 9)
    assert np.isfinite(summary).all()
    assert abs(affine[0, 0]) < 0.1 and affine[1, 0] < -1.5


def test_direct_and_fitted_decoder_profiles() -> None:
    pytest.importorskip("torch")
    from rogii.cli.emission_decoder import _predict_profile

    frame = pd.DataFrame(
        {
            "cut_id": ["a"] * 5 + ["b"] * 5,
            "posterior_mean": np.linspace(-2, 2, 10),
            "true_offset": np.linspace(-1, 1, 10),
            **{f"affine_{index}": np.linspace(-2, 2, 10) for index in range(1)},
            **{f"summary_{index}": np.linspace(-2, 2, 10) + index for index in range(9)},
        }
    )
    direct = _predict_profile({"kind": "direct", "weight": 0.5}, frame)
    affine = _predict_profile({"kind": "affine", "alpha": 10.0}, frame, train_frame=frame)
    summary = _predict_profile({"kind": "summary", "alpha": 100.0}, frame, train_frame=frame)
    assert np.allclose(direct, frame["posterior_mean"] * 0.5)
    assert np.isfinite(affine).all() and np.isfinite(summary).all()


def test_stage23c_notebook_is_clean_and_compiles() -> None:
    path = ROOT / "notebooks" / "590_run_stage23c_oof_decoder.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "'-m','rogii.cli.emission_decoder'" in text
    assert "stage23b_learned_emission_full_v001" in text
    assert "PYTHONPATH" in text and "_driver.log" in text
    assert payload["metadata"]["stage23c"]["submission"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")
