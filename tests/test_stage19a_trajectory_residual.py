from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rogii.cli.trajectory_residual import _cut_feature_record, _hidden_target_invariance, main
from rogii.cli.trajectory_package import main as package_main
from rogii.models.trajectory_residual import (
    COEFFICIENT_COLUMNS,
    apply_residual_coefficients,
    fit_residual_coefficients,
    residual_basis,
)


def _frames(rows: int = 240) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = np.arange(rows)
    md = index.astype(float) * 10.0
    z = 900.0 - 0.12 * md
    u = 11000.0 + 0.007 * md + 0.000002 * md**2
    horizontal = pd.DataFrame({
        "well_id": "well0001", "row_index": index, "MD": md,
        "X": 1000.0 + md, "Y": 2000.0 + 0.2 * md, "Z": z,
        "GR": 75.0 + 10.0 * np.sin(index / 8.0), "TVT": u - z,
    })
    tvt = np.linspace(9800.0, 12200.0, 500)
    typewell = pd.DataFrame({"TVT": tvt, "GR": 75.0 + 10.0 * np.sin(tvt / 80.0)})
    return horizontal, typewell


def test_residual_basis_and_fit_recover_smooth_correction() -> None:
    rows = 400
    base = np.linspace(10_000.0, 10_040.0, rows)
    coefficients = np.array([5.0, -3.0, 2.0])
    truth = base + residual_basis(rows, 80.0) @ coefficients
    fitted = fit_residual_coefficients(base, truth, ramp_rows=80.0, ridge=1e-8)
    assert len(fitted) == len(COEFFICIENT_COLUMNS)
    assert np.allclose(fitted, coefficients, atol=1e-5)
    prediction = apply_residual_coefficients(base, fitted, weight=1.0, cap_ft=50.0, ramp_rows=80.0)
    assert np.sqrt(np.mean(np.square(prediction - truth))) < 1e-5


def test_correction_is_bounded_and_preserves_finite_output() -> None:
    base = np.full(100, 1000.0)
    prediction = apply_residual_coefficients(
        base, np.array([100.0, 100.0, 100.0]), weight=1.0, cap_ft=8.0, ramp_rows=20.0
    )
    assert np.isfinite(prediction).all()
    assert np.max(np.abs(prediction - base)) <= 8.0


def test_stage19_features_ignore_hidden_suffix_tvt() -> None:
    horizontal, typewell = _frames()
    cut = 100
    base = horizontal["TVT"].to_numpy(float)[cut:] - 4.0
    first = _cut_feature_record(horizontal, typewell, cut, base, {"typewell_shift_grid_ft": [-10, 0, 10]})
    changed = horizontal.copy()
    changed.loc[changed.index[cut:], "TVT"] += 999.0
    second = _cut_feature_record(changed, typewell, cut, base, {"typewell_shift_grid_ft": [-10, 0, 10]})
    assert first == second
    assert _hidden_target_invariance(horizontal, typewell, cut, base, {}) is True


def test_stage19a_cli_writes_crossfitted_artifacts(tmp_path: Path) -> None:
    data_dir, artifacts = tmp_path / "data", tmp_path / "artifacts"
    train = data_dir / "train"
    stage16, stage17a, stage17b = artifacts / "stage16", artifacts / "stage17a", artifacts / "stage17b"
    for path in (train, stage16, stage17a, stage17b):
        path.mkdir(parents=True, exist_ok=True)
    cuts, assignments, replay, selector = [], [], [], []
    for well_index in range(12):
        well_id = f"w{well_index:07d}"
        horizontal, typewell = _frames(160)
        horizontal = horizontal.drop(columns=["well_id", "row_index"])
        horizontal["X"] += well_index * 500.0
        horizontal["Y"] += well_index * 250.0
        horizontal["GR"] += well_index * 0.7
        horizontal["TVT"] += well_index * 2.0
        horizontal["TVT_input"] = np.where(np.arange(len(horizontal)) < 64, horizontal["TVT"], np.nan)
        typewell["GR"] += well_index * 0.5
        horizontal.to_csv(train / f"{well_id}__horizontal_well.csv", index=False)
        typewell.to_csv(train / f"{well_id}__typewell.csv", index=False)
        assignments.append({
            "well_id": well_id, "fold": well_index % 3, "spatial_fold": (well_index // 2) % 3,
            "branch_group_fold": (well_index // 3) % 3, "branch_group": well_index,
            "typewell_gr_mean": float(typewell["GR"].mean()), "typewell_gr_std": float(typewell["GR"].std()),
            "typewell_gr_q10": float(typewell["GR"].quantile(0.1)),
            "typewell_gr_q50": float(typewell["GR"].quantile(0.5)),
            "typewell_gr_q90": float(typewell["GR"].quantile(0.9)),
            "typewell_tvt_min": float(typewell["TVT"].min()),
            "typewell_tvt_max": float(typewell["TVT"].max()),
            "typewell_tvt_span": float(typewell["TVT"].max() - typewell["TVT"].min()),
        })
        for cut_index in (64, 96):
            cut_id = f"{well_id}__cut{cut_index}"
            eligible = (well_index + cut_index) % 2 == 0
            cuts.append({
                "cut_id": cut_id, "well_id": well_id, "cut_index": cut_index,
                "requested_fraction": cut_index / len(horizontal), "evaluation_role": "primary",
                "replay_eligible": eligible,
            })
            truth = horizontal["TVT"].to_numpy(float)[cut_index:]
            x = np.linspace(0.0, 1.0, len(truth))
            base = truth - (2.0 + 0.3 * well_index) * (1.0 - np.exp(-np.arange(1, len(truth) + 1) / 30.0)) * (0.5 + x)
            target = replay if eligible else selector
            target.extend({
                "cut_id": cut_id, "row_index": row_index, "y_pred": prediction,
            } for row_index, prediction in zip(range(cut_index, len(horizontal)), base, strict=True))
    pd.DataFrame(assignments).to_parquet(stage16 / "well_assignments.parquet", index=False)
    pd.DataFrame(cuts).to_parquet(stage17b / "cut_report.parquet", index=False)
    pd.DataFrame(replay).to_parquet(stage17a / "replay_predictions.parquet", index=False)
    pd.DataFrame(selector).to_parquet(stage17b / "selector_predictions.parquet", index=False)
    (stage17b / "summary.json").write_text(
        json.dumps({"stage16b_manifest_sha256": "synthetic-stage16"}), encoding="utf-8"
    )
    config = {
        "seed": 42, "provenance": {"stage16b_manifest_sha256": "synthetic-stage16"},
        "features": {"typewell_shift_grid_ft": [-10, 0, 10]},
        "target": {"ramp_rows": 30.0, "ridge": 0.1},
        "model": {"max_iter": 20, "min_samples_leaf": 2, "max_leaf_nodes": 7, "max_depth": 3},
        "profile": {"weight": 0.5, "cap_ft": 16.0, "ramp_rows": 30.0},
        "diagnostics": {"weights": [0.5], "caps_ft": [16.0]},
        "validation": {
            "n_typewell_folds": 3, "bootstrap_resamples": 30,
            "minimum_standard_gain": 0.0, "minimum_improved_fold_fraction": 0.0,
        },
    }
    config_path = tmp_path / "stage19.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    main([
        "--config", str(config_path), "--stage16b-run", str(stage16),
        "--stage17a-run", str(stage17a), "--stage17b-run", str(stage17b),
        "--data-dir", str(data_dir), "--artifact-dir", str(artifacts), "--run-id", "stage19-test",
    ])
    run = artifacts / "stage19-test"
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["stage19a_complete"] is True
    assert summary["cuts"] == 24
    assert summary["wells"] == 12
    assert summary["hidden_target_invariance"] is True
    assert set(summary["family_reports"]) == {"fold", "spatial_fold", "typewell_fold", "branch_group_fold"}
    assert summary["runtime_contract"]["predicted_values_per_well"] == 3
    for name in [
        "cut_features.parquet", "fold_coefficient_oof.parquet",
        "spatial_fold_coefficient_oof.parquet", "typewell_fold_coefficient_oof.parquet",
        "branch_group_fold_coefficient_oof.parquet", "standard_cut_metrics.parquet",
        "profile_report.parquet", "well_assignments.parquet", "config.yaml", "environment.json",
    ]:
        assert (run / name).is_file(), name

    summary["promoted_to_stage19b"] = True
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    package_config = {
        "ensemble_seeds": [42, 52],
        "features": {"typewell_shift_grid_ft": [-10, 0, 10]},
        "model": {"max_iter": 15, "min_samples_leaf": 2, "max_leaf_nodes": 7, "max_depth": 3},
        "benchmark": {
            "requested_fraction": 0.4, "estimated_hidden_wells": 200,
            "maximum_estimated_seconds": 1_000_000.0,
        },
    }
    package_config_path = tmp_path / "stage19b.yaml"
    package_config_path.write_text(yaml.safe_dump(package_config), encoding="utf-8")
    package_main([
        "--config", str(package_config_path), "--stage19a-run", str(run),
        "--stage17a-run", str(stage17a), "--stage17b-run", str(stage17b),
        "--data-dir", str(data_dir), "--artifact-dir", str(artifacts), "--run-id", "stage19b-test",
    ])
    package_run = artifacts / "stage19b-test"
    package_summary = json.loads((package_run / "summary.json").read_text(encoding="utf-8"))
    assert package_summary["stage19b_complete"] is True
    assert package_summary["promoted_to_stage19c"] is True
    assert package_summary["model_count"] == 6
    assert package_summary["benchmark"]["wells"] == 12
    assert package_summary["benchmark"]["maximum_feature_difference"] <= 1e-10
    assert all(package_summary["gates"].values())
    assert (package_run / "stage19b_trajectory_bundle.zip").is_file()
    assert (package_run / "stage19b_trajectory_bundle" / "manifest.json").is_file()
