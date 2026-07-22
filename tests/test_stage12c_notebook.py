from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "290_run_stage12c_spatial_kbest_lattice.ipynb"


def test_stage12c_notebook_is_standalone_spatial_kbest_audit() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    metadata = payload["metadata"]["stage12c"]
    assert metadata["submission"] is False
    assert metadata["spatial_folds"] == 6
    assert metadata["typewell_folds"] == 5
    assert metadata["k_best"] == 16
    assert "drive.mount('/content/drive')" in text
    assert "rogii.cli.emission_lattice" in text
    assert "rogii.cli.emission_tcn" in text
    assert "LIMIT_WELLS = None" in text
    assert "--resume" in text
    assert "promoted_to_all_train_alignment" in text
    assert "submission.csv" not in text


def test_stage12c_notebook_code_compiles() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
