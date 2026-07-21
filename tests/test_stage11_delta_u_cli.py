from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rogii.cli.delta_u import main
from rogii.cli.delta_u_gate import main as gate_main


def _write_well(root: Path, index: int) -> None:
    well_id = f"w{index:07d}"
    row = np.arange(120)
    md = row.astype(float) * 12.0
    z = 900.0 - 0.15 * md
    u = 11000.0 + index * 3.0 + (0.008 + index * 0.0002) * md + 0.000002 * md**2
    tvt = u - z
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": index * 500.0 + 0.8 * md,
            "Y": index * 250.0 + 0.1 * md,
            "Z": z,
            "GR": 70.0 + index + 8.0 * np.sin(row / (6.0 + index * 0.1)),
            "TVT": tvt,
            "TVT_input": np.where(row < 45, tvt, np.nan),
        }
    )
    horizontal.to_csv(root / f"{well_id}__horizontal_well.csv", index=False)
    type_tvt = np.linspace(10000.0, 12000.0, 180)
    pd.DataFrame(
        {"TVT": type_tvt, "GR": 65.0 + index * 2.0 + 9.0 * np.sin(type_tvt / (30.0 + index))}
    ).to_csv(root / f"{well_id}__typewell.csv", index=False)


def test_stage11_cli_writes_audited_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)
    for index in range(12):
        _write_well(train_dir, index)
    config = {
        "seed": 42,
        "multicut": {"fractions": [0.4, 0.65], "min_prefix_rows": 20, "min_suffix_rows": 20},
        "model": {
            "max_iter": 15,
            "min_samples_leaf": 4,
            "max_leaf_nodes": 7,
            "regional": {"k_neighbors": 3},
        },
        "evaluation": {"max_eval_rows_per_cut": 32, "correction_cap_ft": 20.0},
        "validation": {"n_splits": 3, "bootstrap_resamples": 50},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"
    main(
        [
            "--config",
            str(config_path),
            "--run-id",
            "stage11_test",
            "--data-dir",
            str(data_dir),
            "--artifact-dir",
            str(artifact_dir),
        ]
    )
    run = artifact_dir / "stage11_test"
    summary = json.loads((run / "gate_summary.json").read_text(encoding="utf-8"))
    assert summary["n_wells"] == 12
    assert summary["n_cuts"] == 24
    assert summary["hidden_target_invariance"]["passed"] is True
    assert len(summary["feature_schema_hash"]) == 64
    assert {"standard_gain", "spatial_gain", "typewell_gain"}.issubset(summary["gates"])
    for name in [
        "multicut_records.parquet",
        "well_folds.parquet",
        "fold_coefficient_oof.parquet",
        "spatial_fold_coefficient_oof.parquet",
        "typewell_fold_coefficient_oof.parquet",
        "oof.parquet",
        "per_well_metrics.parquet",
        "config.yaml",
        "environment.json",
    ]:
        assert (run / name).is_file(), name

    gate_config = {
        "seed": 42,
        "grid": {"weights": [0.35, 1.0], "correction_caps_ft": [20.0]},
        "selection": {
            "minimum_selection_gain": 0.0,
            "inner_fold_tolerance": 100.0,
            "inference_fold_tolerance": 100.0,
            "tail_tolerance": 10.0,
        },
        "validation": {
            "minimum_standard_gain": 0.0,
            "minimum_improved_fold_fraction": 0.0,
            "bootstrap_resamples": 50,
            "cut_fractions": [0.4, 0.65],
        },
    }
    gate_config_path = tmp_path / "gate_config.yaml"
    gate_config_path.write_text(yaml.safe_dump(gate_config), encoding="utf-8")
    gate_main(
        [
            "--config",
            str(gate_config_path),
            "--stage11-run",
            str(run),
            "--run-id",
            "stage11c_test",
            "--data-dir",
            str(data_dir),
            "--artifact-dir",
            str(artifact_dir),
        ]
    )
    gate_run = artifact_dir / "stage11c_test"
    gate_summary = json.loads((gate_run / "gate_summary.json").read_text(encoding="utf-8"))
    assert gate_summary["n_specs"] == 2
    assert len(gate_summary["cut_report"]) == 2
    assert (gate_run / "nested_oof.parquet").is_file()
    assert (gate_run / "spec_report.parquet").is_file()
    assert (gate_run / "cut_report.parquet").is_file()
