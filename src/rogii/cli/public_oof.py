from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.evaluation.gates import evaluate_candidate_gates
from rogii.evaluation.metrics import evaluate_predictions
from rogii.models.public_residual import (
    DEFAULT_PUBLIC_FEATURES,
    apply_public_delta_postprocess,
    build_public_residual_features,
    crossfit_positive_ridge,
    crossfit_public_residual,
    fit_full_public_residual_models,
)
from rogii.models.spatial import make_spatial_blocks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct the public pretrained OOF stack and gate a residual correction"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--public-artifacts-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit-wells", type=int)
    return parser


def _trainer_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("models/lightgbm-*/*.pkl")) + sorted(
        root.glob("models/catboost-*/*.pkl")
    )
    if len(paths) != 5:
        raise FileNotFoundError(
            f"Expected 5 serialized public trainers under {root / 'models'}, found {len(paths)}"
        )
    return paths


def _load_frame(csv_path: Path, requested_features: list[str]) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    required = ["well", "id", "target", "last_known_tvt", "pf_ancc", "md_since"]
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(f"Public train.csv is missing required columns: {missing}")
    selected = list(dict.fromkeys(required + [name for name in requested_features if name in header]))
    dtype: dict[str, object] = {name: np.float32 for name in selected if name not in {"well", "id"}}
    dtype.update({"well": "string", "id": "string"})
    print(f"reading {csv_path} with {len(selected)} of {len(header)} columns", flush=True)
    frame = pd.read_csv(csv_path, usecols=selected, dtype=dtype, low_memory=False)
    frame = frame.rename(columns={"well": "well_id"})
    frame["well_id"] = frame["well_id"].astype(str)
    frame["id"] = frame["id"].astype(str)
    frame["row_index"] = pd.to_numeric(
        frame["id"].str.rsplit("_", n=1).str[-1], errors="raise"
    ).astype(np.int32)
    frame["MD"] = frame["md_since"].astype(np.float32)
    return frame


def _load_model_oof(paths: list[Path], mask: np.ndarray, expected_rows: int) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for path in paths:
        print(f"loading public trainer: {path}", flush=True)
        try:
            trainer = joblib.load(path)
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Loading public trainers requires lightgbm, catboost, and koolbox. "
                "Run the supplied Colab notebook, which installs them before this command."
            ) from error
        values = np.asarray(trainer.oof_preds, dtype=np.float64).reshape(-1)
        if len(values) != expected_rows:
            raise ValueError(f"{path}: OOF rows {len(values)} != train.csv rows {expected_rows}")
        predictions.append(values[mask])
        del trainer
    return np.column_stack(predictions)


def _well_coordinates(data_dir: Path, wells: list[str]) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for well_id in wells:
        path = data_dir / "train" / f"{well_id}__horizontal_well.csv"
        horizontal = pd.read_csv(path, usecols=["X", "Y", "TVT_input"])
        known = horizontal[horizontal["TVT_input"].notna()]
        anchor = known.iloc[-1] if len(known) else horizontal.iloc[0]
        records.append({"well_id": well_id, "x": float(anchor["X"]), "y": float(anchor["Y"])})
    return pd.DataFrame.from_records(records)


def _prediction_frame(frame: pd.DataFrame, prediction: np.ndarray, fold: np.ndarray) -> pd.DataFrame:
    output = frame[["id", "well_id", "row_index", "MD"]].copy()
    output["Z"] = frame["z"].to_numpy(dtype=np.float32) if "z" in frame else 0.0
    output["fold"] = np.asarray(fold, dtype=np.int16)
    output["y_true"] = (
        frame["last_known_tvt"].to_numpy(dtype=np.float64)
        + frame["target"].to_numpy(dtype=np.float64)
    )
    output["y_pred"] = np.asarray(prediction, dtype=np.float64)
    return output


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    public_config = dict(config.get("public_stack", {}))
    residual_config = dict(config.get("residual", {}))
    model_config = dict(config.get("model", {}))
    gate_config = dict(config.get("gates", {}))
    seed = int(config.get("seed", 42))
    n_splits = int(public_config.get("n_splits", 5))
    model_seeds = [int(value) for value in config.get("model_seeds", [42, 43])]
    if not model_seeds:
        raise ValueError("model_seeds must not be empty")

    output_dir = args.artifact_dir.resolve() / args.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    public_root = args.public_artifacts_dir.resolve()
    csv_path = public_root / "data" / "train.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing public artifact train.csv: {csv_path}")
    requested_features = [str(value) for value in residual_config.get("features", DEFAULT_PUBLIC_FEATURES)]
    full_frame = _load_frame(csv_path, requested_features)
    expected_rows = len(full_frame)
    selected_wells = full_frame["well_id"].drop_duplicates()
    if args.limit_wells is not None:
        if args.limit_wells < n_splits:
            raise ValueError("limit-wells must be at least the number of folds")
        selected_wells = selected_wells.iloc[: args.limit_wells]
    mask = full_frame["well_id"].isin(selected_wells).to_numpy()
    frame = full_frame.loc[mask].reset_index(drop=True)
    del full_frame

    base_model_oof = _load_model_oof(_trainer_paths(public_root), mask, expected_rows)
    ridge_delta, folds = crossfit_positive_ridge(
        base_model_oof,
        frame["target"].to_numpy(dtype=np.float64),
        frame["well_id"],
        n_splits=n_splits,
        alpha=float(public_config.get("ridge_alpha", 1.6602834637650032)),
        tol=float(public_config.get("ridge_tol", 0.0005030247295617308)),
    )
    frame["fold"] = folds
    base_prediction = apply_public_delta_postprocess(
        frame,
        ridge_delta,
        alpha=float(public_config.get("alpha", 1.0)),
        tau_md=float(public_config.get("tau_md", 85.0)),
        pf_weight=float(public_config.get("pf_weight", 0.09)),
    )
    features = build_public_residual_features(
        frame,
        base_model_oof,
        base_prediction,
        requested_columns=requested_features,
        max_rows_per_well=int(residual_config.get("max_rows_per_well", 256)),
    )
    target_clip = float(residual_config.get("target_clip", 60.0))
    standard_residual = crossfit_public_residual(
        features,
        folds,
        model_config,
        model_seeds,
        target_clip,
    )

    coordinates = _well_coordinates(args.data_dir.resolve(), selected_wells.astype(str).tolist())
    well_folds = coordinates.copy()
    n_spatial_blocks = min(int(residual_config.get("n_spatial_blocks", 6)), len(well_folds))
    well_folds["spatial_fold"] = make_spatial_blocks(well_folds, n_spatial_blocks, seed)
    spatial_fold_map = well_folds.set_index("well_id")["spatial_fold"]
    spatial_folds = frame["well_id"].map(spatial_fold_map).to_numpy(dtype=np.int16)
    spatial_residual = crossfit_public_residual(
        features,
        spatial_folds,
        model_config,
        model_seeds,
        target_clip,
    )

    weight = float(residual_config.get("correction_weight", 0.25))
    if not 0.0 <= weight <= 1.0:
        raise ValueError("correction_weight must be between zero and one")
    baseline = _prediction_frame(frame, base_prediction, folds)
    candidate = _prediction_frame(frame, base_prediction + weight * standard_residual, folds)
    candidate["base_y_pred"] = base_prediction
    candidate["residual_pred"] = standard_residual
    candidate["correction_weight"] = weight
    spatial_baseline = _prediction_frame(frame, base_prediction, spatial_folds)
    spatial_candidate = _prediction_frame(
        frame, base_prediction + weight * spatial_residual, spatial_folds
    )
    spatial_candidate["base_y_pred"] = base_prediction
    spatial_candidate["residual_pred"] = spatial_residual
    spatial_candidate["correction_weight"] = weight

    gate_summary = evaluate_candidate_gates(
        baseline,
        candidate,
        spatial_baseline,
        spatial_candidate,
        minimum_standard_gain=float(gate_config.get("minimum_standard_gain", 0.05)),
        minimum_spatial_gain=float(gate_config.get("minimum_spatial_gain", 0.02)),
        minimum_improved_fold_fraction=float(
            gate_config.get("minimum_improved_fold_fraction", 0.8)
        ),
        bootstrap_resamples=int(gate_config.get("bootstrap_resamples", 2000)),
        seed=seed,
    )

    weight_grid: list[dict[str, float]] = []
    base_metrics, _ = evaluate_predictions(baseline)
    for grid_weight in [float(value) for value in residual_config.get("weight_grid", [0.1, 0.2, 0.25, 0.35, 0.5])]:
        grid_frame = _prediction_frame(frame, base_prediction + grid_weight * standard_residual, folds)
        grid_metrics, _ = evaluate_predictions(grid_frame)
        weight_grid.append(
            {
                "weight": grid_weight,
                "pooled_rmse": float(grid_metrics["pooled_rmse"]),
                "pooled_rmse_delta": float(grid_metrics["pooled_rmse"])
                - float(base_metrics["pooled_rmse"]),
                "well_rmse_p90": float(grid_metrics["well_rmse_p90"]),
                "worst_10pct_sse_share": float(grid_metrics["worst_10pct_sse_share"]),
            }
        )

    full_models = fit_full_public_residual_models(
        features, model_config, model_seeds, target_clip
    )
    for model_seed, model in zip(model_seeds, full_models, strict=True):
        with (output_dir / f"public_residual_full_seed_{model_seed}.pkl").open("wb") as handle:
            pickle.dump(model, handle)

    baseline.to_parquet(output_dir / "base_oof.parquet", index=False)
    candidate.to_parquet(output_dir / "oof.parquet", index=False)
    spatial_candidate.to_parquet(output_dir / "spatial_oof.parquet", index=False)
    well_folds.to_parquet(output_dir / "spatial_wells.parquet", index=False)
    metrics, well_metrics = evaluate_predictions(candidate)
    well_metrics.to_parquet(output_dir / "per_well_metrics.parquet", index=False)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "gate_summary.json", gate_summary)
    write_json(output_dir / "weight_grid.json", weight_grid)
    write_json(output_dir / "feature_columns.json", features.columns)
    write_json(output_dir / "environment.json", environment_report())
    package_manifest = {
        "schema_version": "rogii_public_residual_v1",
        "base": "ravaghi_public_artifact_oof_stack",
        "n_rows": len(frame),
        "n_wells": int(frame["well_id"].nunique()),
        "feature_columns": "feature_columns.json",
        "models": [f"public_residual_full_seed_{value}.pkl" for value in model_seeds],
        "correction_weight": weight,
        "target_clip": target_clip,
        "promoted": bool(gate_summary["promoted"]),
        "conditional_spatial_audit": True,
        "conditional_spatial_note": (
            "The residual corrector is refit by spatial block, but the fixed public base OOF "
            "was produced with ordinary well GroupKFold."
        ),
    }
    write_json(output_dir / "model_manifest.json", package_manifest)
    resolved = dict(config)
    resolved["resolved"] = {
        "public_artifacts_dir": str(public_root),
        "data_dir": str(args.data_dir.resolve()),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "run_id": args.run_id,
        "limit_wells": args.limit_wells,
    }
    write_yaml(output_dir / "config.yaml", resolved)
    print(json.dumps(gate_summary, ensure_ascii=False, indent=2))
    print("weight grid:")
    print(pd.DataFrame(weight_grid).to_string(index=False))
    print(f"run artifacts: {output_dir}")


if __name__ == "__main__":
    main()
