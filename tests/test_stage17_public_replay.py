from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from rogii.cli.public_replay import _summarize_cuts, main, replay_is_target_safe


def test_replay_requires_public_prefix_to_be_contained() -> None:
    assert replay_is_target_safe(20, 20)
    assert replay_is_target_safe(20, 30)
    assert not replay_is_target_safe(30, 20)


def test_role_summary_uses_public_only_where_eligible() -> None:
    cuts = pd.DataFrame({
        "evaluation_role": ["primary", "primary"],
        "replay_eligible": [True, False],
        "suffix_rows": [4, 4],
        "baseline_sse": [16.0, 16.0],
        "public_sse": [4.0, float("nan")],
        "hybrid_sse": [4.0, 16.0],
    })
    report = _summarize_cuts(cuts, "primary")
    assert report["cut_coverage"] == 0.5
    assert report["row_coverage"] == 0.5
    assert report["eligible_baseline_rmse"] == 2.0
    assert report["eligible_public_rmse"] == 1.0
    assert report["hybrid_rmse"] < report["baseline_rmse"]


def test_cli_replays_aligned_public_suffix(tmp_path) -> None:
    data_dir, stage16, public_run, artifacts = (
        tmp_path / "data", tmp_path / "stage16", tmp_path / "public", tmp_path / "artifacts"
    )
    (data_dir / "train").mkdir(parents=True)
    stage16.mkdir(); public_run.mkdir(); artifacts.mkdir()
    manifest_rows, public_rows = [], []
    for fold in range(5):
        well = f"w{fold}"
        tvt = np.arange(10, dtype=float) + fold
        pd.DataFrame({"MD": np.arange(10, dtype=float), "TVT": tvt}).to_csv(
            data_dir / "train" / f"{well}__horizontal_well.csv", index=False
        )
        manifest_rows.append({
            "well_id": well, "cut_id": f"{well}__cut4", "cut_index": 4,
            "suffix_rows": 6, "requested_fraction": 0.4,
            "evaluation_role": "primary", "fold": fold,
        })
        for row in range(2, 10):
            public_rows.append({
                "id": f"{well}_{row}", "well_id": well, "row_index": row,
                "fold": fold, "y_true": tvt[row], "y_pred": tvt[row] - 0.25,
            })
    pd.DataFrame(manifest_rows).to_parquet(stage16 / "pseudo_test_manifest.parquet", index=False)
    (stage16 / "summary.json").write_text(json.dumps({"manifest_sha256": "frozen"}))
    pd.DataFrame(public_rows).to_parquet(public_run / "base_oof.parquet", index=False)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "provenance": {"stage16b_manifest_sha256": "frozen", "target_tolerance": 1e-5},
        "gates": {"minimum_primary_row_coverage": 0.35, "minimum_eligible_gain": 0.05,
                  "minimum_improved_folds": 4},
    }))
    main([
        "--config", str(config), "--stage16b-run", str(stage16),
        "--public-oof-run", str(public_run), "--data-dir", str(data_dir),
        "--artifact-dir", str(artifacts), "--run-id", "run",
    ])
    summary = json.loads((artifacts / "run" / "summary.json").read_text())
    assert summary["stage17a_complete"]
    assert summary["promoted_to_selector_replay"]
    assert summary["role_report"]["primary"]["row_coverage"] == 1.0
    assert len(pd.read_parquet(artifacts / "run" / "replay_predictions.parquet")) == 30
