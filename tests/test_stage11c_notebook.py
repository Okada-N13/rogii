from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "260_run_stage11c_delta_u_robust_gate.ipynb"


def test_stage11c_notebook_is_standalone_nested_audit() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    metadata = payload["metadata"]["stage11c"]
    assert metadata["submission"] is False
    assert metadata["reuses_stage11"] is True
    assert metadata["nested_selection"] is True
    assert metadata["absolute_tail_gate"] is True
    assert "drive.mount('/content/drive')" in text
    assert "rogii-delta-u-gate" in text
    assert "stage11c_delta_u_robust_gate_full_v001" in text
    assert "selected_inference_spec" in text
    assert "cut_report" in text
    assert "submission.csv" not in text


def test_stage11c_notebook_code_compiles() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
