from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_viterbi_decoder_prefers_continuous_path() -> None:
    pytest.importorskip("torch")
    from rogii.cli.emission_path_decoder import viterbi_offsets

    offsets = np.array([-1.0, 0.0, 1.0])
    logits = np.array(
        [
            [0.0, 4.0, 0.0],
            [0.0, 3.0, 3.2],
            [0.0, 4.0, 0.0],
        ]
    )
    path = viterbi_offsets(logits, offsets, transition_weight=1.0, anchor_weight=0.0)
    assert np.array_equal(path, np.zeros(3))


def test_smoothing_and_profile_decoding_are_finite() -> None:
    pytest.importorskip("torch")
    from rogii.cli.emission_path_decoder import _centered_smooth, decode_profile

    values = np.array([0.0, 0.0, 9.0, 0.0, 0.0])
    smooth = _centered_smooth(values, 3)
    assert smooth[2] < values[2]
    logits = np.tile(np.array([[0.0, 3.0, 0.0]]), (5, 1))
    decoded = decode_profile(
        {"kind": "posterior_median", "decoder_weight": 0.25},
        logits,
        np.array([-1.0, 0.0, 1.0]),
    )
    assert np.isfinite(decoded).all() and np.allclose(decoded, 0.0)


def test_stage25a_notebook_is_clean_and_does_not_use_reserve() -> None:
    path = ROOT / "notebooks" / "620_run_stage25a_temporal_path_decoder.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "rogii.cli.emission_path_decoder" in text
    assert "stage24a_scaled_ordinal_emission_full_v001" in text
    assert payload["metadata"]["stage25a"]["submission"] is False
    assert payload["metadata"]["stage25a"]["reserved_confirmation_used"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")
