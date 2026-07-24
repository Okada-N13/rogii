from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rogii.cli.residual_profile_lock import _audit_profile


ROOT = Path(__file__).resolve().parents[1]


def test_uniformly_better_predeclared_profile_is_eligible() -> None:
    rows = []
    for index in range(60):
        rows.append(
            {
                "well_id": f"well_{index:03d}",
                "suffix_rows": 100,
                "base_sse": 10_000.0,
                "candidate_sse_locked": 7_225.0,
                "stage16_fold": index % 5,
                "spatial_fold": index % 6,
                "typewell_fold": (index // 2) % 5,
                "branch_group_fold": (index // 3) % 5,
                "requested_fraction": (0.22, 0.26, 0.30, 0.34)[index % 4],
            }
        )
    audit = _audit_profile(
        pd.DataFrame(rows),
        "locked",
        {"bootstrap_resamples": 200, "minimum_gain": 0.10},
        seed=42,
    )
    assert audit["eligible"] is True
    assert audit["rmse_delta"] < -0.10
    assert audit["bootstrap_95pct"][1] < 0.0
    assert all(audit["gates"].values())


def test_stage32a_notebook_is_clean_cpu_only_and_reserved_safe() -> None:
    path = ROOT / "notebooks" / "690_run_stage32a_tcn_profile_lock.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "rogii-residual-profile-lock" in text
    assert "stage32a_tcn_profile_lock.yaml" in text
    assert "stage31a_tcn_uncertainty_full_v001" in text
    assert "torch.cuda" not in text
    assert payload["metadata"]["stage32a"]["reserved_confirmation_used"] is False
    assert payload["metadata"]["stage32a"]["training"] is False
    assert payload["metadata"]["stage32a"]["submission"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")
