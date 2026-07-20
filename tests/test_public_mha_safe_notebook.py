from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "95_kaggle_public_mha_safe.ipynb"


def test_public_mha_safe_notebook_keeps_mha_and_removes_probes() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "DELTA midhedge" in text
    assert "PF_BIMODAL_STATS" in text
    assert "probe canary" not in text
    assert "LB PROBE v4" not in text
    assert "_BC_SHIFT" not in text
    assert "Guarded contact override v2" not in text
    assert "ROGII_PROBE" not in text


def test_public_mha_safe_notebook_has_one_final_artifact_audit() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    metadata = payload["metadata"]["stage6_safe_build"]
    assert metadata["source"] == "canqiang/rogii-det-mha140sep4"
    assert metadata["probe_bias_removed"] is True
    assert "removed_ambiguous_csvs" in text
    assert "submission_sha256" in text
    assert "id_order_matches_sample" in text
