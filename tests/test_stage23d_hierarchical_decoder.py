from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _frame() -> pd.DataFrame:
    rows = 60
    x = np.linspace(-2.0, 2.0, rows)
    true = np.where(x < -0.5, -8.0, np.where(x > 0.5, 8.0, 0.0))
    return pd.DataFrame(
        {
            "cut_id": np.repeat(["a", "b", "c"], rows // 3),
            "true_offset": true,
            **{
                f"summary_{index}": x + 0.01 * index
                for index in range(9)
            },
        }
    )


def test_hierarchical_decoder_is_finite_and_directional() -> None:
    from rogii.cli.emission_hierarchical_decoder import (
        fit_hierarchical_decoder,
        predict_hierarchical_decoder,
    )

    frame = _frame()
    profile = {
        "classifier_c": 0.1,
        "magnitude_alpha": 10.0,
        "move_threshold_ft": 3.0,
        "magnitude_cap_ft": 16.0,
        "move_power": 1.0,
        "direction_power": 1.0,
        "decoder_weight": 0.75,
    }
    fitted = fit_hierarchical_decoder(frame, profile)
    prediction = predict_hierarchical_decoder(fitted, frame, profile)
    assert np.isfinite(prediction).all()
    assert prediction[:10].mean() < 0
    assert prediction[-10:].mean() > 0
    assert np.max(np.abs(prediction)) <= 12.0


def test_stage23d_notebook_is_clean_and_cpu_only() -> None:
    path = ROOT / "notebooks" / "600_run_stage23d_hierarchical_decoder.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "rogii-emission-hierarchical-decoder" in text
    assert "stage23c_oof_decoder_full_v001" in text
    assert payload["metadata"]["stage23d"]["submission"] is False
    assert payload["metadata"]["stage23d"]["gpu_required"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")
