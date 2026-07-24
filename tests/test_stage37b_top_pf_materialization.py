from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rogii.cli.pseudo_private_top_pf import main


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "750_run_stage37b_top_pf_materialization.ipynb"


def _source(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def test_stage37b_materializes_only_training_and_design_and_resumes(tmp_path: Path) -> None:
    data, artifacts = tmp_path / "data", tmp_path / "artifacts"
    train, stage17, stage37 = data / "train", artifacts / "stage17", artifacts / "stage37"
    for path in (train, stage17, stage37):
        path.mkdir(parents=True)
    manifest_rows, replay_rows = [], []
    for index, (well_id, role) in enumerate(
        (("train001", "training"), ("train002", "training"), ("design01", "design_validation"),
         ("confirm1", "confirmation_locked"))
    ):
        rows, cut = 60, 20 + index
        position = np.arange(rows)
        tvt = 10_000.0 + index * 5.0 + position * 0.3
        horizontal = pd.DataFrame({
            "MD": position * 10.0, "X": position * 4.0 + index * 20.0,
            "Y": position * 2.0, "Z": 900.0 - position,
            "GR": 70.0 + np.sin(position / 4.0), "TVT": tvt,
        })
        typewell = pd.DataFrame({
            "TVT": np.linspace(9_900.0, 10_200.0, 200),
            "GR": 70.0 + np.sin(np.arange(200) / 8.0),
        })
        horizontal.to_csv(train / f"{well_id}__horizontal_well.csv", index=False)
        typewell.to_csv(train / f"{well_id}__typewell.csv", index=False)
        cut_id = f"{well_id}__cut{cut}"
        manifest_rows.append({
            "cut_id": cut_id, "well_id": well_id, "cut_index": cut,
            "requested_fraction": cut / rows, "suffix_rows": rows - cut,
            "benchmark_role": role, "fold": index % 2, "spatial_fold": index % 2,
            "typewell_fold": index % 2, "branch_group_fold": index % 2,
        })
        if role != "confirmation_locked":
            replay_rows.extend({
                "cut_id": cut_id, "row_index": int(row), "y_pred": float(tvt[row] - 1.0)
            } for row in range(cut, rows))
    pd.DataFrame(manifest_rows).to_parquet(stage37 / "pseudo_private_manifest.parquet", index=False)
    pd.DataFrame(replay_rows).to_parquet(stage17 / "replay_predictions.parquet", index=False)
    (stage37 / "summary.json").write_text(json.dumps({
        "pseudo_private_manifest_sha256": "synthetic",
        "promoted_to_stage37b_top_pf_replay": True,
    }), encoding="utf-8")
    config = {
        "provenance": {"stage37a_manifest_sha256": "synthetic"},
        "selector": {
            "particles": 4, "seeds": 1, "seed_base": 0, "maximum_tracking_steps": 20,
            "typewell_grid_step": 1.0, "gr_sigma_multiplier": 1.3,
        },
        "top_pf_proxy": {
            "ridge_weight": 0.3, "selector_weight": 0.7, "projection_degree": 2,
            "projection_blend_weight": 0.75, "final_sp45_weight": 0.6,
        },
        "storage": {"cuts_per_chunk": 2},
        "audit": {"invariance_cuts": 2},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    arguments = [
        "--config", str(config_path), "--stage17a-run", str(stage17),
        "--stage37a-run", str(stage37), "--data-dir", str(data),
        "--artifact-dir", str(artifacts), "--run-id", "stage37b",
    ]
    main(arguments)
    first = json.loads((artifacts / "stage37b" / "summary.json").read_text(encoding="utf-8"))
    assert first["promoted_to_stage38_retrieval_v2"] is True
    assert first["materialized_cuts"] == 3
    assert first["prediction_rows"] == sum(row["suffix_rows"] for row in manifest_rows[:3])
    assert first["confirmation_wells_materialized"] == []
    assert first["confirmation_target_columns_read"] is False
    assert all(first["gates"].values())
    main(arguments)
    second = json.loads((artifacts / "stage37b" / "summary.json").read_text(encoding="utf-8"))
    assert all(row["status"] == "reused" for row in second["chunk_report"])
    assert second["gates"]["hidden_target_invariance"] is True


def test_stage37b_notebook_is_clean_resumable_and_compiles() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join(_source(cell) for cell in payload["cells"])
    assert "rogii-pseudo-private-top-pf" in text
    assert "stage37b_top_pf_materialization_v001" in text
    assert "Do not delete" in text
    assert "subprocess.run(['uv','run','rogii-pseudo-private-top-pf'" in text
    assert payload["metadata"]["stage37b"]["resumable_chunks"] is True
    assert payload["metadata"]["stage37b"]["confirmation_target_unread"] is True
    for index, cell in enumerate(payload["cells"]):
        assert cell.get("outputs", []) == []
        if cell.get("cell_type") == "code":
            ast.parse(_source(cell), filename=f"cell_{index}")
