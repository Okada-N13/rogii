from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "230_kaggle_v599_a130_frontier_safe.ipynb"


def _payload_and_sources() -> tuple[dict, list[str], str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in payload["cells"]]
    return payload, sources, "\n".join(sources)


def test_v599_safe_keeps_reported_frontier_recipe() -> None:
    payload, sources, text = _payload_and_sources()
    metadata = payload["metadata"]["v599_frontier_safe_build"]
    assert metadata["reported_public_score"] == 6.768
    assert metadata["sp45_blend_weight"] == 0.60
    assert metadata["visible_prefix_alpha_multiplier"] == 1.30
    assert metadata["model_package_gated_max_weight"] == 0.00425
    assert metadata["branch_strength"] == 1.0
    assert metadata["branch_cap"] == 3.0
    assert metadata["branch_skip_existing"] is True

    assert "SUBMISSION_PROFILE = 'vp_balanced_modelpkg_005'" in text
    assert "SP45_BLEND_WEIGHT = float(_profile['sp45_blend_weight'])" in text
    assert "alpha * 1.30" in text
    assert "_BH_STRENGTH = 1.00" in text
    assert "_BH_CAP = 3.00" in text

    gold = next(i for i, source in enumerate(sources) if "Visible-prefix calibration overlay" in source)
    model_package = next(
        i for i, source in enumerate(sources)
        if source.lstrip().startswith("# Optional saved-model correction")
    )
    branch = next(i for i, source in enumerate(sources) if "Guarded PF seed-branch midpoint hedge" in source)
    final_audit = next(i for i, source in enumerate(sources) if "V599_FRONTIER_SAFE_AUDIT" in source)
    assert gold < model_package < branch < final_audit


def test_v599_safe_removes_target_transfer_and_probe_paths() -> None:
    payload, _, text = _payload_and_sources()
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

    metadata = payload["metadata"]["v599_frontier_safe_build"]
    assert metadata["same_well_target_transfer_removed"] is True
    assert metadata["final_single_artifact"] is True
    assert "removed_ambiguous_csvs" in text
    assert "leaderboard_bias_removed" in text
    assert "lb_probe_removed" in text


def test_v599_safe_code_cells_compile() -> None:
    payload, _, _ = _payload_and_sources()
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") != "code":
            continue
        ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
