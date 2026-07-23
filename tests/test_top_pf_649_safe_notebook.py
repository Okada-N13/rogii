from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "470_kaggle_top_pf_a130_branch_safe.ipynb"
PARENT = ROOT / "notebooks" / "230_kaggle_v599_a130_frontier_safe.ipynb"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def test_top_pf_safe_keeps_only_hidden_active_reported_changes() -> None:
    payload, parent = _load(NOTEBOOK), _load(PARENT)
    assert len(payload["cells"]) == len(parent["cells"])
    changed = [
        i for i, (new, old) in enumerate(zip(payload["cells"], parent["cells"]))
        if _source(new) != _source(old)
    ]
    assert changed == [0, 36, 48, 50]
    text = "\n".join(_source(cell) for cell in payload["cells"])
    assert "10., 60.)) * 1.3" in text
    assert "_BH_STRENGTH = 0.60" in text
    assert "_BH_CAP = 2.00" in text
    assert "_BH_SKIP_EXISTING = False" in text
    assert "top_pf_a130_branch_conservative_safe" in text


def test_top_pf_safe_removes_public_placeholder_target_paths() -> None:
    payload = _load(NOTEBOOK)
    text = "\n".join(_source(cell) for cell in payload["cells"])
    forbidden = [
        "tvt_from_contacts", "_gold_contact_candidate", "_gold_reapply_guarded_contact_override",
        "RUN_OVERLAP_DRY_RUN_PROBE = True", "RUN_GUARDED_OVERLAP_OVERRIDE = True",
        "hw_tr['TVT']", 'hw_tr["TVT"]', "LB PROBE", "_BC_SHIFT",
    ]
    for marker in forbidden:
        assert marker not in text
    metadata = payload["metadata"]["top_pf_649_safe_build"]
    assert metadata["reported_public_score"] == 6.49
    assert metadata["same_well_target_transfer_removed"] is True
    assert metadata["stage18_included"] is False


def test_top_pf_safe_cells_compile_and_outputs_are_clear() -> None:
    payload = _load(NOTEBOOK)
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            assert cell.get("outputs", []) == []
            ast.parse(_source(cell), filename=f"cell_{index}")
