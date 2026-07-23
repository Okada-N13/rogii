from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "480_run_stage19a_trajectory_residual.ipynb"


def _source(cell: dict) -> str:
    value = cell.get("source", [])
    return "".join(value) if isinstance(value, list) else str(value)


def test_stage19a_notebook_is_standalone_and_non_submission() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join(_source(cell) for cell in payload["cells"])
    assert "git', 'clone'" in text
    assert "uv', 'sync', '--frozen'" in text
    assert "rogii-trajectory-residual" in text
    assert "stage16b_testlike_validation_full_v003" in text
    assert "stage17_public_replay_full_v002" in text
    assert "stage17b_selector_replay_full_v001" in text
    assert payload["metadata"]["stage19a"]["submission"] is False
    assert payload["metadata"]["stage19a"]["predicted_values_per_well"] == 3


def test_stage19a_notebook_code_compiles_and_has_no_outputs() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            assert cell.get("outputs", []) == []
            ast.parse(_source(cell), filename=f"cell_{index}")
