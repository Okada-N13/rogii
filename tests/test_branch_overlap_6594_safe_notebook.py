from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "240_kaggle_branch_overlap_6594_safe.ipynb"
PARENT = ROOT / "notebooks" / "230_kaggle_v599_a130_frontier_safe.ipynb"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def test_branch_overlap_matches_the_reported_6594_delta() -> None:
    payload = _load(NOTEBOOK)
    metadata = payload["metadata"]["branch_overlap_6594_safe_build"]
    assert metadata["reported_public_score"] == 6.594
    assert metadata["sanitized_parent"] == PARENT.name
    assert metadata["branch_strength"] == 0.60
    assert metadata["branch_cap"] == 2.0
    assert metadata["branch_skip_existing"] is False

    text = "\n".join(_source(cell) for cell in payload["cells"])
    assert "_BH_STRENGTH = 0.60" in text
    assert "_BH_CAP = 2.00" in text
    assert "_BH_SKIP_EXISTING = False" in text
    assert "BRANCH_OVERLAP_6594_SAFE_AUDIT" in text


def test_only_branch_and_audit_differ_from_sanitized_parent() -> None:
    payload = _load(NOTEBOOK)
    parent = _load(PARENT)
    assert len(payload["cells"]) == len(parent["cells"])
    changed = [
        index for index, (new_cell, old_cell) in enumerate(zip(payload["cells"], parent["cells"]))
        if _source(new_cell) != _source(old_cell)
    ]
    assert changed == [0, 48, 50]


def test_branch_overlap_safe_paths_and_syntax() -> None:
    payload = _load(NOTEBOOK)
    text = "\n".join(_source(cell) for cell in payload["cells"])
    forbidden = [
        "tvt_from_contacts",
        "_gold_contact_candidate",
        "_gold_reapply_guarded_contact_override",
        "RUN_OVERLAP_DRY_RUN_PROBE = True",
        "RUN_GUARDED_OVERLAP_OVERRIDE = True",
        "hw_tr['TVT']",
        'hw_tr["TVT"]',
        "LB PROBE",
        "_BC_SHIFT",
    ]
    for marker in forbidden:
        assert marker not in text
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse(_source(cell), filename=f"cell_{index}")
