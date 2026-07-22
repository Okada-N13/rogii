from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "460_kaggle_v599_stage18_ranked_retrieval.ipynb"
PARENT = ROOT / "notebooks" / "230_kaggle_v599_a130_frontier_safe.ipynb"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def test_stage18e_is_inserted_after_frozen_branch_and_before_final_audit() -> None:
    payload = _load(NOTEBOOK)
    sources = [_source(cell) for cell in payload["cells"]]
    branch = next(i for i, source in enumerate(sources) if "Guarded PF seed-branch midpoint hedge" in source)
    retrieval = next(i for i, source in enumerate(sources) if "STAGE18E_TEST_AUDIT" in source)
    final_audit = next(i for i, source in enumerate(sources) if "V599_FRONTIER_SAFE_AUDIT" in source)
    assert branch < retrieval < final_audit

    metadata = payload["metadata"]["stage18e_ranked_retrieval"]
    assert metadata["base_public_lb"] == 6.685
    assert metadata["blend_weight"] == 0.20
    assert metadata["selected_donors"] == 4
    assert metadata["same_well_target_transfer_removed"] is True
    assert metadata["internet"] is False
    assert metadata["package_manifest_sha256"] == (
        "7bddc1914f3d046b678dbb8f5d1cc17427b03bc85c1a06d1f2088cbe68d3935d"
    )


def test_stage18e_preserves_parent_and_enforces_submission_audits() -> None:
    payload, parent = _load(NOTEBOOK), _load(PARENT)
    assert len(payload["cells"]) == len(parent["cells"]) + 1
    text = "\n".join(_source(cell) for cell in payload["cells"])
    assert "len(_s18_statuses) != 3" in text
    assert "any(status != 'applied'" in text
    assert "same_well_target_transfer_removed" in json.dumps(payload["metadata"])
    assert "v599_a130_branch_conservative_stage18_ranked_retrieval" in text
    assert all(cell.get("outputs", []) == [] for cell in payload["cells"] if cell.get("cell_type") == "code")


def test_stage18e_code_cells_compile() -> None:
    payload = _load(NOTEBOOK)
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse(_source(cell), filename=f"cell_{index}")
