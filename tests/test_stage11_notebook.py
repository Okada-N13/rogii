from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "250_run_stage11_multicut_delta_u.ipynb"


def test_stage11_notebook_is_standalone_and_non_submission() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    metadata = payload["metadata"]["stage11"]
    assert metadata["submission"] is False
    assert metadata["public_predictions"] is False
    assert metadata["hidden_target_invariance"] is True
    assert "drive.mount('/content/drive')" in text
    assert "git', 'clone'" in text
    assert "git', '-C', str(repo_dir), 'pull', '--ff-only'" in text
    assert "uv', 'sync', '--frozen'" in text
    assert "rogii-delta-u" in text
    assert "LIMIT_WELLS = None" in text
    assert "stage11_multicut_delta_u_full_v001" in text
    assert "submission.csv" not in text


def test_stage11_notebook_code_compiles() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
