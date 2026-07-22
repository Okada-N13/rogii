from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "280_run_stage12b_learned_emission_tcn.ipynb"


def test_stage12b_notebook_is_standalone_gpu_oof() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    metadata = payload["metadata"]["stage12b"]
    assert metadata["submission"] is False
    assert metadata["states"] == 61
    assert metadata["oof_folds"] == 5
    assert "drive.mount('/content/drive')" in text
    assert "rogii.cli.emission_tcn" in text
    assert "rogii-delta-u" in text
    assert "rogii-raw-ncc" in text
    assert "LIMIT_WELLS = None" in text
    assert "--device', 'cuda" in text
    assert "--resume" in text
    assert "promoted_to_spatial_emission_audit" in text
    assert "submission.csv" not in text


def test_stage12b_notebook_code_compiles() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
