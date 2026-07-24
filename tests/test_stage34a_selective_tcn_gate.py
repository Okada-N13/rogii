from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.cli.residual_selective_gate import apply_hard_gate


ROOT = Path(__file__).resolve().parents[1]


def test_hard_gate_uses_only_full_or_base_sse() -> None:
    frame = pd.DataFrame(
        {
            "predicted_gate": [0.2, 0.6, 0.9],
            "base_sse": [100.0, 100.0, 100.0],
            "full_sse": [80.0, 70.0, 60.0],
        }
    )
    result = apply_hard_gate(frame, 0.6)
    assert np.array_equal(result.to_numpy(), np.array([100.0, 70.0, 60.0]))


def test_stage34a_notebook_is_clean_cpu_only_and_reserved_safe() -> None:
    path = ROOT / "notebooks" / "710_run_stage34a_selective_tcn_gate.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "rogii-residual-selective-gate" in text
    assert "stage34a_selective_tcn_gate.yaml" in text
    assert "stage33a_tcn_cut_benefit_gate_full_v001" in text
    assert "torch.cuda" not in text
    assert payload["metadata"]["stage34a"]["reserved_confirmation_used"] is False
    assert payload["metadata"]["stage34a"]["training"] is False
    assert payload["metadata"]["stage34a"]["submission"] is False
    assert payload["metadata"]["stage34a"]["tcn_inference"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")
