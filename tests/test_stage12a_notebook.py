from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "270_run_stage12a_raw_ncc_benchmark.ipynb"


def test_stage12a_notebook_is_standalone_benchmark() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    metadata = payload["metadata"]["stage12a"]
    assert metadata["submission"] is False
    assert metadata["surface_spec"] == "w075_cap50"
    assert metadata["raw_ncc"] is True
    assert metadata["hidden_target_invariance"] is True
    assert "drive.mount('/content/drive')" in text
    assert "rogii-raw-ncc" in text
    assert "LIMIT_WELLS = None" in text
    assert "promoted_to_learned_emission" in text
    assert "variant_report.parquet" in text
    assert "submission.csv" not in text


def test_stage12a_notebook_code_compiles() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
