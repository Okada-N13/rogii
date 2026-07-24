from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rogii.models.residual_uncertainty import uncertainty_shrunk_residual


ROOT = Path(__file__).resolve().parents[1]


def test_confidence_gate_keeps_agreement_and_shrinks_disagreement() -> None:
    predictions = np.array([[2.0, 2.0], [2.0, -2.0], [2.0, 1.0]])
    residual, gate = uncertainty_shrunk_residual(predictions, kind="confidence", power=1.0)
    assert gate[0] > 0.99
    assert gate[1] < gate[0]
    assert residual[0] > residual[1]


def test_sign_gate_zeroes_inconsistent_rows() -> None:
    predictions = np.array([[1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [1.0, -1.0]])
    residual, gate = uncertainty_shrunk_residual(
        predictions, kind="sign_agreement", minimum_agreement=0.6
    )
    assert gate[0] == 1.0 and residual[0] == 1.0
    assert gate[1] == 0.0 and residual[1] == 0.0


def test_stage31a_notebook_is_clean_and_reserved_safe() -> None:
    path = ROOT / "notebooks" / "680_run_stage31a_tcn_uncertainty.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "'-m','rogii.cli.residual_tcn_uncertainty'" in text
    assert "stage31a_tcn_uncertainty.yaml" in text
    assert "stage30a_residual_tcn_full_v001" in text
    assert payload["metadata"]["stage31a"]["reserved_confirmation_used"] is False
    assert payload["metadata"]["stage31a"]["submission"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")

