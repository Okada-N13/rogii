from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from rogii.cli.stage36_retrieval_benchmark import projected_hidden_runtime


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "730_run_stage36_parallel_retrieval_benchmark.ipynb"


def _source(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def test_runtime_projection_is_conservative_and_batched() -> None:
    assert projected_hidden_runtime(30.0, 20.0, 3, 200, 4) == 520.0
    assert projected_hidden_runtime(30.0, 20.0, 3, 3, 4) == 20.0
    with pytest.raises(ValueError):
        projected_hidden_runtime(0.0, 20.0, 3, 200, 4)


def test_stage36_notebook_contract_and_code() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join(_source(cell) for cell in payload["cells"])
    assert "stage36_parallel_retrieval_package_v001" in text
    assert "stage36_parallel_retrieval_benchmark_v001" in text
    assert "'--workers','4'" in text
    assert "'--hidden-wells','200'" in text
    assert "'--runtime-gate-seconds','600'" in text
    assert payload["metadata"]["accelerator"] == "CPU"
    assert payload["metadata"]["stage36"]["exact_byte_parity"] is True
    for index, cell in enumerate(payload["cells"]):
        assert cell.get("outputs", []) == []
        if cell.get("cell_type") == "code":
            ast.parse(_source(cell), filename=f"cell_{index}")


def test_stage18_package_and_kaggle_notebook_require_parallel_v4() -> None:
    package_source = (ROOT / "src" / "rogii" / "cli" / "stage18_package.py").read_text(encoding="utf-8")
    kaggle = json.loads(
        (ROOT / "notebooks" / "460_kaggle_v599_stage18_ranked_retrieval.ipynb").read_text(encoding="utf-8")
    )
    kaggle_text = "\n".join(_source(cell) for cell in kaggle["cells"])
    assert '"package_version": 4' in package_source
    assert '"well_workers": 4' in package_source
    assert "package_version', 0)) < 4" in kaggle_text
    assert "v004 parallel package is required" in kaggle_text
