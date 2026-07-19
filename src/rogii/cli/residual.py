from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.data.loading import discover_horizontal_wells, well_id_from_path
from rogii.evaluation.metrics import evaluate_predictions
from rogii.models.residual_sequence import (
    build_residual_features,
    make_residual_model,
    sampled_training_data,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-fit a Stage 3 residual sequence model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--limit-wells", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    training = dict(config.get("training", {}))
    model_config = dict(config.get("model", {}))
    output_dir = args.artifact_dir.resolve() / args.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_oof = pd.read_parquet(args.base_run / "oof.parquet")
    if args.limit_wells is not None:
        selected_wells = base_oof["well_id"].drop_duplicates().iloc[: args.limit_wells]
        base_oof = base_oof[base_oof["well_id"].isin(selected_wells)].copy()
    paths = discover_horizontal_wells(args.data_dir, "train")
    path_map = {well_id_from_path(path): path for path in paths}
    features = build_residual_features(
        base_oof,
        path_map,
        windows=[int(value) for value in training.get("sequence_windows", [9, 33, 129])],
        max_rows_per_well=int(training.get("max_rows_per_well", 256)),
    )
    frame = features.frame
    folds = sorted(int(value) for value in frame["fold"].unique())
    if len(folds) < 2:
        raise ValueError("Cross-fitting requires at least two folds")
    residual_prediction = np.empty(len(frame), dtype=np.float64)
    target_clip = float(training.get("target_clip", 80.0))
    seed = int(config.get("seed", 42))
    model_seeds = [int(value) for value in config.get("model_seeds", [seed])]
    if not model_seeds:
        raise ValueError("model_seeds must be non-empty")
    for fold in folds:
        x_train, y_train, weights = sampled_training_data(features, fold, target_clip)
        valid = frame["fold"].to_numpy(dtype=int) == fold
        x_valid = frame.loc[valid, features.columns].to_numpy(dtype=np.float32)
        fold_predictions: list[np.ndarray] = []
        for model_seed in model_seeds:
            model = make_residual_model(model_config, model_seed + fold)
            model.fit(x_train, y_train, sample_weight=weights)
            fold_predictions.append(model.predict(x_valid))
            with (output_dir / f"model_fold_{fold}_seed_{model_seed}.pkl").open("wb") as handle:
                pickle.dump(model, handle)
        residual_prediction[valid] = np.vstack(fold_predictions).mean(axis=0)
        print(f"cross-fit fold {fold}: train={len(x_train)} valid={int(valid.sum())}", flush=True)

    x_train, y_train, weights = sampled_training_data(features, None, target_clip)
    for model_seed in model_seeds:
        full_model = make_residual_model(model_config, model_seed + 10_000)
        full_model.fit(x_train, y_train, sample_weight=weights)
        with (output_dir / f"model_full_seed_{model_seed}.pkl").open("wb") as handle:
            pickle.dump(full_model, handle)

    correction_weight = float(training.get("correction_weight", 0.1))
    if not 0.0 <= correction_weight <= 1.0:
        raise ValueError("correction_weight must be between zero and one")
    predictions = frame[
        [
            "id",
            "well_id",
            "row_index",
            "MD",
            "Z",
            "fold",
            "base_y_pred",
            "y_true",
            "pf_seed_std",
            "pf_gr_sigma",
            "pf_log_likelihood_spread",
        ]
    ].copy()
    predictions["residual_pred"] = residual_prediction
    predictions["correction_weight"] = correction_weight
    predictions["y_pred"] = predictions["base_y_pred"] + correction_weight * residual_prediction
    metrics, well_metrics = evaluate_predictions(predictions)
    predictions.to_parquet(output_dir / "oof.parquet", index=False)
    well_metrics.to_parquet(output_dir / "per_well_metrics.parquet", index=False)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "environment.json", environment_report())
    write_json(output_dir / "feature_columns.json", features.columns)
    config["resolved"] = {
        "base_run": str(args.base_run.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "run_id": args.run_id,
        "limit_wells": args.limit_wells,
    }
    write_yaml(output_dir / "config.yaml", config)
    (output_dir / "run.log").write_text(
        f"wells={predictions['well_id'].nunique()}\nrows={len(predictions)}\n"
        f"pooled_rmse={metrics['pooled_rmse']:.8f}\n",
        encoding="utf-8",
    )
    print(f"pooled RMSE: {metrics['pooled_rmse']:.6f}")
    print(f"run artifacts: {output_dir}")


if __name__ == "__main__":
    main()
