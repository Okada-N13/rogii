from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "80_run_stage6_public7_reproduction.ipynb"


def test_stage6_notebook_is_valid_and_self_contained() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "kaitofukami/rogii-public-7-061-exact-reproduction" in source
    assert "kaggle kernels" not in source  # subprocess arguments avoid shell/token leakage
    assert "['kaggle', 'kernels', 'pull', '-p'" in source
    assert "id_order_matches_sample" in source
    assert "submission_sha256" in source
    assert "git', 'clone'" in source


def test_stage6_keeps_public_score_separate_from_honest_oof() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    all_text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "public_leaderboard_positive_control" in all_text
    assert "not an honest 773-well OOF score" in all_text
