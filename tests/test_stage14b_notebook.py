from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "320_run_stage14b_extended_residual_gate.ipynb"


def test_stage14b_notebook_is_fast_absolute_tail_audit() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    metadata = payload["metadata"]["stage14b"]
    assert metadata["submission"] is False
    assert metadata["cpu_only"] is True
    assert metadata["retrain"] is False
    assert metadata["absolute_tail_gate"] is True
    assert "drive.mount('/content/drive')" in text
    assert "rogii-emission-residual-gate" in text
    assert "LIMIT_WELLS = None" in text
    assert "standard_absolute_tail_delta" in text
    assert "promoted_to_full_residual_training" in text
    assert "submission.csv" not in text


def test_stage14b_notebook_code_compiles() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
