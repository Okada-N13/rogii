from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "300_run_stage13_emission_uncertainty_gate.ipynb"


def test_stage13_notebook_is_cpu_validation_only() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    metadata = payload["metadata"]["stage13"]
    assert metadata["submission"] is False
    assert metadata["cpu_only"] is True
    assert metadata["strict_holdout_profiles"] is True
    assert "drive.mount('/content/drive')" in text
    assert "rogii-emission-uncertainty" in text
    assert "LIMIT_WELLS = None" in text
    assert "promoted_to_all_train_uncertainty_model" in text
    assert "submission.csv" not in text


def test_stage13_notebook_code_compiles() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
