from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "310_run_stage14_crossfit_emission_residual.ipynb"


def test_stage14_notebook_is_cpu_strict_crossfit() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    metadata = payload["metadata"]["stage14"]
    assert metadata["submission"] is False
    assert metadata["cpu_only"] is True
    assert metadata["strict_family_crossfit"] is True
    assert metadata["resume"] is True
    assert "drive.mount('/content/drive')" in text
    assert "rogii-emission-residual" in text
    assert "LIMIT_WELLS = None" in text
    assert "--resume" in text
    assert "promoted_to_full_residual_training" in text
    assert "submission.csv" not in text


def test_stage14_notebook_code_compiles() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
