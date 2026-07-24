from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_scaled_manifest_reserves_disjoint_wells(tmp_path: Path) -> None:
    from rogii.cli.scaled_emission_manifest import main

    stage17 = tmp_path / "stage17"
    validation = tmp_path / "validation"
    artifacts = tmp_path / "artifacts"
    stage17.mkdir()
    validation.mkdir()
    rows = []
    for fold in range(5):
        for index in range(130):
            well = f"f{fold}_{index:03d}"
            rows.append(
                {
                    "cut_id": f"{well}__cut",
                    "well_id": well,
                    "stage16_fold": fold,
                    "requested_fraction": 0.30,
                    "evaluation_role": "primary",
                    "replay_eligible": True,
                    "original_public_cut_index": 80,
                    "cut_index": 100,
                }
            )
    validation_rows = []
    for fold in range(5):
        well = f"validation_{fold}"
        cut_id = f"{well}__cut"
        rows.append(
            {
                "cut_id": cut_id,
                "well_id": well,
                "stage16_fold": fold,
                "requested_fraction": 0.30,
                "evaluation_role": "primary",
                "replay_eligible": True,
                "original_public_cut_index": 80,
                "cut_index": 100,
            }
        )
        validation_rows.append({"cut_id": cut_id})
    pd.DataFrame(rows).to_parquet(stage17 / "cut_report.parquet", index=False)
    pd.DataFrame(validation_rows).to_parquet(
        validation / "confidence_cut_report.parquet", index=False
    )
    main(
        [
            "--stage17a-run", str(stage17),
            "--design-validation-run", str(validation),
            "--artifact-dir", str(artifacts),
            "--run-id", "manifest",
            "--training-wells-per-fold", "100",
            "--confirmation-wells-per-fold", "24",
        ]
    )
    summary = json.loads((artifacts / "manifest" / "summary.json").read_text())
    assert summary["training_wells"] == 500
    assert summary["confirmation_wells"] == 120
    assert all(not values for values in summary["overlaps"].values())


def test_soft_ordinal_loss_prefers_nearby_state() -> None:
    torch = pytest.importorskip("torch")
    from rogii.models.emission_tcn import _loss

    offsets = torch.arange(-2, 3, dtype=torch.float32)
    target = torch.tensor([[2]], dtype=torch.long)
    costs = torch.zeros((1, 1, 1, 5), dtype=torch.float32)
    close = torch.tensor([[[0.0, 1.0, 4.0, 1.0, 0.0]]])
    wrong = torch.tensor([[[4.0, 1.0, 0.0, 1.0, 0.0]]])
    config = {
        "ordinal_sigma_ft": 1.0,
        "expected_offset_weight": 1.0,
        "hard_negative_weight": 0.0,
    }
    assert _loss(close, costs, target, config, offsets) < _loss(
        wrong, costs, target, config, offsets
    )


def test_stage24a_notebook_is_clean_and_uses_reserved_split() -> None:
    path = ROOT / "notebooks" / "610_run_stage24a_scaled_ordinal_emission.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "rogii-scaled-emission-manifest" in text
    assert "stage24a_scaled_ordinal_emission.yaml" in text
    assert "stage24a_scaled_emission_manifest_v002" in text
    assert "training_cut_ids.parquet" in text
    assert payload["metadata"]["stage24a"]["submission"] is False
    assert payload["metadata"]["stage24a"]["reserved_confirmation_wells"] == 120
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")
