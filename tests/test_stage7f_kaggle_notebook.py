from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "150_kaggle_public_robust_blend.ipynb"


def _payload_and_text() -> tuple[dict, str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    return payload, text


def test_stage7f_uses_promoted_branch_local_blend() -> None:
    payload, text = _payload_and_text()
    metadata = payload["metadata"]["stage7f_public_blend"]
    assert metadata["branch"] == "package_postprocessed"
    assert metadata["branch_weight"] == 0.4
    assert metadata["effective_weight_before_mha"] == 0.066
    assert "MODEL_PACKAGE_BRANCH_WEIGHT = 0.4" in text
    assert "sub_1 = _mp_validate_submission_ids" in text
    assert "0.3 * x['tvt_1'] + 0.7 * x['tvt_2']" in text
    assert "_SELECTED_SP45_WEIGHT = 0.55" in text
    assert "w_fleongg = 1.0 - float(w_sp45)" in text


def test_stage7f_keeps_exact_package_inference_and_safe_audit() -> None:
    _, text = _payload_and_text()
    assert "model_package_manifest.json" in text
    assert "class TCNRegressor" in text
    assert "_mp_apply_delta_postprocess" in text
    assert "_mp_apply_savgol" in text
    assert "stage7f_public_blend_audit.json" in text
    assert "submission_sha256" in text
    assert "removed_ambiguous_csvs" in text
    assert "MODEL_PACKAGE_GATED" not in text
    assert "LB PROBE" not in text
    assert "probe canary" not in text
