from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "210_kaggle_public_mha_stage10c.ipynb"
SNIPPET = ROOT / "notebooks" / "snippets" / "stage10c_alignment_cell.py"


def test_stage10c_snippet_compiles_and_has_promoted_profile() -> None:
    text = SNIPPET.read_text(encoding="utf-8")
    ast.parse(text)
    assert "_A_BRANCH_WEIGHT = 0.20" in text
    assert "_A_CORRECTION_CAP = 8.0" in text
    assert "_A_MIN_PREFIX_CORRELATION = 0.30" in text
    assert "stage10c_alignment_audit.json" in text
    assert "horizontal['TVT']" not in text


def test_stage10c_notebook_inserts_alignment_before_mha_and_keeps_safety() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in payload["cells"]]
    text = "\n".join(sources)
    alignment_index = next(i for i, source in enumerate(sources) if "STAGE10C_ALIGNMENT_AUDIT" in source)
    blend_index = next(i for i, source in enumerate(sources) if "0.3 * x['tvt_1'] + 0.7 * x['tvt_2']" in source)
    assert alignment_index < blend_index
    assert "DELTA midhedge" in text
    assert "removed_ambiguous_csvs" in text
    assert "probe canary" not in text
    assert "LB PROBE v4" not in text
    metadata = payload["metadata"]["stage10c_alignment"]
    assert metadata["profile"] == "prefix030_cap8"
    assert metadata["base_notebook"] == "95_kaggle_public_mha_safe.ipynb"


def test_stage10c_preserves_every_safe_baseline_cell_except_title() -> None:
    baseline = json.loads((ROOT / "notebooks" / "95_kaggle_public_mha_safe.ipynb").read_text(encoding="utf-8"))
    candidate = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    alignment_index = next(
        i for i, cell in enumerate(candidate["cells"])
        if "STAGE10C_ALIGNMENT_AUDIT" in "".join(cell.get("source", []))
    )
    retained = candidate["cells"][:alignment_index] + candidate["cells"][alignment_index + 1 :]
    assert len(retained) == len(baseline["cells"])
    assert retained[1:] == baseline["cells"][1:]
