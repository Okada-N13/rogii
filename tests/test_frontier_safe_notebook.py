from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "220_kaggle_public_frontier_safe.ipynb"


def _payload_and_sources() -> tuple[dict, list[str], str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in payload["cells"]]
    return payload, sources, "\n".join(sources)


def test_frontier_safe_keeps_promoted_public_components() -> None:
    payload, sources, text = _payload_and_sources()
    metadata = payload["metadata"]["frontier_safe_build"]
    assert metadata["reported_public_score"] == 6.858
    assert metadata["profile"] == "mha250sep2_gold_conservative"
    assert '_MH_ALPHA, _MH_MINMASS, _MH_SEPLO, _MH_SEPHI, _MH_CAP = 2.5, 0.22, 2.0, 40.0, 4.0' in text
    assert 'os.environ["ROGII_GOLD_PROFILE"] = "conservative"' in text
    gold = next(i for i, source in enumerate(sources) if "Gold visible-prefix calibration overlay" in source)
    midpoint = next(i for i, source in enumerate(sources) if source.lstrip().startswith("# === DELTA midhedge"))
    contract = next(i for i, source in enumerate(sources) if "Final hidden-set contract transaction" in source)
    cleanup = next(i for i, source in enumerate(sources) if "FRONTIER_SAFE_AUDIT" in source)
    assert gold < midpoint < contract < cleanup


def test_frontier_safe_removes_lb_and_same_well_target_paths() -> None:
    payload, _, text = _payload_and_sources()
    forbidden = [
        "_BC_SHIFT", "ROGII_PROBE", "LB PROBE", "Guarded contact override",
        "_gold_contact_candidate", "_gold_reapply_guarded_contact_override",
        "tvt_from_contacts", "hw_tr['TVT']", 'hw_tr["TVT"]',
    ]
    for marker in forbidden:
        assert marker not in text
    metadata = payload["metadata"]["frontier_safe_build"]
    assert metadata["global_bias_removed"] is True
    assert metadata["probe_paths_removed"] is True
    assert metadata["same_well_target_transfer_removed"] is True
    assert "removed_ambiguous_csvs" in text


def test_frontier_safe_code_cells_compile() -> None:
    payload, _, _ = _payload_and_sources()
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") != "code":
            continue
        ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
