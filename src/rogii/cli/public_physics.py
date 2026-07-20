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
from rogii.models.public_physics import (
    apply_physics_spec,
    build_prefix_physics_features,
    make_physics_specs,
    nested_select_predictions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nested-CV public OOF physics calibration")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _with_prediction(base: pd.DataFrame, prediction: np.ndarray, folds: np.ndarray) -> pd.DataFrame:
    output = base.copy(deep=False)
    output = output.assign(y_pred=np.asarray(prediction), fold=np.asarray(folds, dtype=np.int16))
    return output


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    physics_config = dict(config.get("physics", {}))
    gate_config = dict(config.get("gates", {}))
    output_dir = args.artifact_dir.resolve() / args.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_run = args.base_run.resolve()
    base = pd.read_parquet(base_run / "base_oof.parquet")
    spatial_wells = pd.read_parquet(base_run / "spatial_wells.parquet")
    standard_folds = base["fold"].to_numpy(dtype=np.int16)
    spatial_map = spatial_wells.set_index("well_id")["spatial_fold"]
    spatial_folds = base["well_id"].map(spatial_map).to_numpy(dtype=np.int16)
    tails = [int(value) for value in physics_config.get("tails", [160, 320, 1000000])]
    degrees = [int(value) for value in physics_config.get("degrees", [1, 2])]
    features = build_prefix_physics_features(base, args.data_dir.resolve(), tails, degrees)
    poly_columns = [name for name in features if name.startswith("poly_u_")]
    specs = make_physics_specs(physics_config, poly_columns)
    minimum_selection_gain = float(physics_config.get("minimum_selection_gain", 0.02))

    standard_prediction, standard_selections, ranking = nested_select_predictions(
        base, features, standard_folds, specs, minimum_selection_gain
    )
    spatial_prediction, spatial_selections, spatial_ranking = nested_select_predictions(
        base, features, spatial_folds, specs, minimum_selection_gain
    )
    baseline = _with_prediction(base, base["y_pred"].to_numpy(), standard_folds)
    candidate = _with_prediction(base, standard_prediction, standard_folds)
    spatial_baseline = _with_prediction(base, base["y_pred"].to_numpy(), spatial_folds)
    spatial_candidate = _with_prediction(base, spatial_prediction, spatial_folds)
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

    full_spec_row = ranking[0]
    full_spec = specs[int(full_spec_row["index"])]
    full_prediction = apply_physics_spec(base, features, full_spec)
    full_frame = _with_prediction(base, full_prediction, standard_folds)
    full_metrics, _ = evaluate_predictions(full_frame)
    package = {
        "schema_version": "rogii_public_physics_v1",
        "promoted": bool(gate["promoted"]),
        "honest_evaluation": "nested well-fold hyperparameter selection",
        "inference_spec": full_spec.to_dict(),
        "inference_spec_oof_rmse": float(full_metrics["pooled_rmse"]),
        "warning": "The inference spec is fit on all OOF; promotion is determined only by nested predictions.",
    }

    candidate.to_parquet(output_dir / "oof.parquet", index=False)
    spatial_candidate.to_parquet(output_dir / "spatial_oof.parquet", index=False)
    features.to_parquet(output_dir / "physics_features.parquet", index=False)
    write_json(output_dir / "gate_summary.json", gate)
    write_json(output_dir / "standard_selections.json", standard_selections)
    write_json(output_dir / "spatial_selections.json", spatial_selections)
    write_json(output_dir / "ranking.json", ranking[:30])
    write_json(output_dir / "spatial_ranking.json", spatial_ranking[:30])
    write_json(output_dir / "physics_manifest.json", package)
    write_json(output_dir / "environment.json", environment_report())
    resolved = dict(config)
    resolved["resolved"] = {"base_run": str(base_run), "data_dir": str(args.data_dir.resolve())}
    write_yaml(output_dir / "config.yaml", resolved)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print("standard nested selections:")
    print(json.dumps(standard_selections, ensure_ascii=False, indent=2))
    print("inference package:")
    print(json.dumps(package, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

