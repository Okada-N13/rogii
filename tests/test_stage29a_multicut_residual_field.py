from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rogii.cli.expanded_residual_manifest import main as manifest_main


ROOT = Path(__file__).resolve().parents[1]


def test_multicut_manifest_preserves_frozen_well_sets(tmp_path: Path) -> None:
    stage17 = tmp_path / "stage17"
    base = tmp_path / "base"
    artifacts = tmp_path / "artifacts"
    stage17.mkdir()
    base.mkdir()
    pd.DataFrame(
        [
            {"cut_id": "a1", "well_id": "a", "stage16_fold": 0, "requested_fraction": 0.22, "evaluation_role": "primary", "cut_index": 10, "replay_eligible": True},
            {"cut_id": "a2", "well_id": "a", "stage16_fold": 0, "requested_fraction": 0.30, "evaluation_role": "primary", "cut_index": 20, "replay_eligible": True},
            {"cut_id": "b1", "well_id": "b", "stage16_fold": 1, "requested_fraction": 0.22, "evaluation_role": "primary", "cut_index": 10, "replay_eligible": True},
            {"cut_id": "c1", "well_id": "c", "stage16_fold": 2, "requested_fraction": 0.22, "evaluation_role": "primary", "cut_index": 10, "replay_eligible": True},
        ]
    ).to_parquet(stage17 / "cut_report.parquet", index=False)
    training = pd.DataFrame(
        [
            {"cut_id": "a1", "well_id": "a", "stage16_fold": 0, "requested_fraction": 0.22, "evaluation_role": "primary", "cut_index": 10},
            {"cut_id": "b1", "well_id": "b", "stage16_fold": 1, "requested_fraction": 0.22, "evaluation_role": "primary", "cut_index": 10},
        ]
    )
    confirmation = pd.DataFrame(
        [{"cut_id": "c1", "well_id": "c", "stage16_fold": 2, "requested_fraction": 0.22, "evaluation_role": "primary", "cut_index": 10}]
    )
    training.to_parquet(base / "training_cut_ids.parquet", index=False)
    confirmation.to_parquet(base / "confirmation_cut_ids.parquet", index=False)
    (base / "summary.json").write_text(
        json.dumps({"training_wells": 500, "confirmation_wells": 120, "overlaps": {}}),
        encoding="utf-8",
    )
    manifest_main(
        [
            "--stage17a-run", str(stage17), "--base-manifest-run", str(base),
            "--artifact-dir", str(artifacts), "--run-id", "run",
        ]
    )
    expanded = pd.read_parquet(artifacts / "run" / "training_cut_ids.parquet")
    assert set(expanded["cut_id"]) == {"a1", "a2", "b1"}
    assert set(expanded["well_id"]) == {"a", "b"}
    assert set(expanded["well_id"]).isdisjoint(confirmation["well_id"])


def test_stage29a_notebook_is_clean_and_reserved_safe() -> None:
    path = ROOT / "notebooks" / "660_run_stage29a_multicut_residual_field.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "rogii-expanded-residual-manifest" in text
    assert "stage29a_multicut_residual_field.yaml" in text
    assert "stage29a_multicut_manifest_v001" in text
    assert payload["metadata"]["stage29a"]["reserved_confirmation_used"] is False
    assert payload["metadata"]["stage29a"]["submission"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")

