from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.evaluation.gates import evaluate_candidate_gates
from rogii.evaluation.metrics import evaluate_predictions
from rogii.models.public_blend import (
    align_package_ground_truth,
    apply_blend_spec,
    load_package_branches,
    make_blend_specs,
    nested_select_blend,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nested blend of verified public OOF branches")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _prediction_frame(base: pd.DataFrame, prediction: np.ndarray, folds: np.ndarray) -> pd.DataFrame:
    return base.assign(y_pred=np.asarray(prediction), fold=np.asarray(folds, dtype=np.int16))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    blend_config = dict(config.get("blend", {}))
    gate_config = dict(config.get("gates", {}))
    output_dir = args.artifact_dir.resolve() / args.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_run = args.base_run.resolve()
    package_root = args.package_dir.resolve()
    base = pd.read_parquet(base_run / "base_oof.parquet")
    package_truth = pd.read_parquet(package_root / "oof" / "train_gt.parquet")
    order, alignment = align_package_ground_truth(base, package_truth)
    branch_files = {
        str(name): str(path) for name, path in dict(blend_config.get("branch_files", {})).items()
    }
    branches = load_package_branches(str(package_root), package_truth, order, branch_files)
    weights = [float(value) for value in blend_config.get("weights", [0.01, 0.02, 0.05, 0.1])]
    specs = make_blend_specs(list(branches), weights)
    minimum_gain = float(blend_config.get("minimum_selection_gain", 0.02))
    standard_folds = base["fold"].to_numpy(dtype=np.int16)
    spatial_wells = pd.read_parquet(base_run / "spatial_wells.parquet")
    spatial_map = spatial_wells.set_index("well_id")["spatial_fold"]
    spatial_folds = base["well_id"].map(spatial_map).to_numpy(dtype=np.int16)
    standard_prediction, standard_selections, ranking = nested_select_blend(
        base, branches, standard_folds, specs, minimum_gain
    )
    spatial_prediction, spatial_selections, spatial_ranking = nested_select_blend(
        base, branches, spatial_folds, specs, minimum_gain
    )
    baseline = _prediction_frame(base, base["y_pred"].to_numpy(), standard_folds)
    candidate = _prediction_frame(base, standard_prediction, standard_folds)
    spatial_baseline = _prediction_frame(base, base["y_pred"].to_numpy(), spatial_folds)
    spatial_candidate = _prediction_frame(base, spatial_prediction, spatial_folds)
    gate = evaluate_candidate_gates(
        baseline,
        candidate,
        spatial_baseline,
        spatial_candidate,
        minimum_standard_gain=float(gate_config.get("minimum_standard_gain", 0.05)),
        minimum_spatial_gain=float(gate_config.get("minimum_spatial_gain", 0.02)),
        minimum_improved_fold_fraction=float(gate_config.get("minimum_improved_fold_fraction", 0.8)),
        bootstrap_resamples=int(gate_config.get("bootstrap_resamples", 2000)),
        seed=int(config.get("seed", 42)),
    )

    branch_metrics = {}
    base_error = base["y_pred"].to_numpy(dtype=np.float64) - base["y_true"].to_numpy(dtype=np.float64)
    for name, prediction in branches.items():
        frame = _prediction_frame(base, prediction, standard_folds)
        metrics, _ = evaluate_predictions(frame)
        error = np.asarray(prediction, dtype=np.float64) - base["y_true"].to_numpy(dtype=np.float64)
        branch_metrics[name] = {
            "pooled_rmse": metrics["pooled_rmse"],
            "error_correlation_with_base": float(np.corrcoef(base_error, error)[0, 1]),
        }

    best_spec = specs[int(ranking[0]["index"])]
    full_prediction = apply_blend_spec(base["y_pred"].to_numpy(), branches, best_spec)
    full_metrics, _ = evaluate_predictions(_prediction_frame(base, full_prediction, standard_folds))
    manifest = {
        "schema_version": "rogii_public_verified_blend_v1",
        "promoted": bool(gate["promoted"]),
        "alignment": alignment,
        "inference_spec": best_spec.to_dict(),
        "inference_spec_oof_rmse": full_metrics["pooled_rmse"],
        "package_manifest": "metadata/model_package_manifest.json",
        "conditional_spatial_audit": True,
        "conditional_spatial_note": (
            "Both public branches are fixed ordinary GroupKFold OOF. Spatial blocks test "
            "blend-weight transfer, but do not retrain package models or their global "
            "self-excluding formation imputer by spatial block."
        ),
        "warning": "Promotion uses nested predictions; the all-OOF inference spec is not itself a gate.",
    }
    candidate.to_parquet(output_dir / "oof.parquet", index=False)
    spatial_candidate.to_parquet(output_dir / "spatial_oof.parquet", index=False)
    write_json(output_dir / "alignment.json", alignment)
    write_json(output_dir / "branch_metrics.json", branch_metrics)
    write_json(output_dir / "gate_summary.json", gate)
    write_json(output_dir / "standard_selections.json", standard_selections)
    write_json(output_dir / "spatial_selections.json", spatial_selections)
    write_json(output_dir / "ranking.json", ranking)
    write_json(output_dir / "spatial_ranking.json", spatial_ranking)
    write_json(output_dir / "blend_manifest.json", manifest)
    write_json(output_dir / "environment.json", environment_report())
    resolved = dict(config)
    resolved["resolved"] = {"base_run": str(base_run), "package_dir": str(package_root)}
    write_yaml(output_dir / "config.yaml", resolved)
    print(json.dumps({"alignment": alignment, "branches": branch_metrics, "gate": gate, "manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
