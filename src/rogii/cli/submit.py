from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.data.loading import (
    discover_horizontal_wells,
    load_horizontal_well,
    load_typewell,
    well_id_from_path,
)
from rogii.data.schema import validate_horizontal_well
from rogii.models.registry import predict_model
from rogii.models.residual_sequence import build_residual_features
from rogii.models.trellis import predict_trellis_correction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an honest Stage 4 Kaggle submission")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage3-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    stage2_config_path = Path(config.get("stage2_config", "configs/experiment/pf_trend_blend.yaml"))
    stage3_config_path = Path(
        config.get("stage3_config", "configs/experiment/stage3_residual_hgb.yaml")
    )
    stage4_config_path = Path(config.get("stage4_config", "configs/experiment/stage4_tail_path.yaml"))
    stage2_config = load_config(stage2_config_path)
    stage3_config = load_config(stage3_config_path)
    stage4_config = load_config(stage4_config_path)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_horizontal_wells(args.data_dir, "test")
    path_map = {well_id_from_path(path): path for path in paths}
    stage2_parts: list[pd.DataFrame] = []
    for path in paths:
        horizontal = load_horizontal_well(path)
        validate_horizontal_well(horizontal, split="test")
        stage2_parts.append(
            predict_model(horizontal, dict(stage2_config["model"]), load_typewell(path))
        )
    stage2 = pd.concat(stage2_parts, ignore_index=True)
    if "y_true" in stage2:
        raise RuntimeError("Test predictions unexpectedly contain y_true")

    training_config = dict(stage3_config.get("training", {}))
    features = build_residual_features(
        stage2,
        path_map,
        windows=[int(value) for value in training_config.get("sequence_windows", [9, 33, 129])],
        max_rows_per_well=int(training_config.get("max_rows_per_well", 256)),
    )
    feature_columns = json.loads((args.stage3_run / "feature_columns.json").read_text())
    if feature_columns != features.columns:
        raise ValueError("Stage 3 model feature columns do not match the current code/config")
    model_seeds = [int(value) for value in stage3_config.get("model_seeds", [42])]
    x_test = features.frame[feature_columns].to_numpy(dtype=np.float32)
    residual_members: list[np.ndarray] = []
    for seed in model_seeds:
        model_path = args.stage3_run / f"model_full_seed_{seed}.pkl"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing Stage 3 full model: {model_path}")
        residual_members.append(_load_pickle(model_path).predict(x_test))
    residual_prediction = np.vstack(residual_members).mean(axis=0)
    correction_weight = float(training_config.get("correction_weight", 0.6))
    stage3 = features.frame.copy()
    stage3["base_y_pred"] = stage3["y_pred"]
    stage3["residual_pred"] = residual_prediction
    stage3["y_pred"] = stage3["base_y_pred"] + correction_weight * residual_prediction

    guard_config = dict(stage4_config.get("guard", {}))
    trellis_config = dict(stage4_config.get("trellis", {}))
    train_oof = pd.read_parquet(args.stage3_run / "oof.parquet")
    train_uncertainty = train_oof.groupby("well_id")["pf_seed_std"].mean()
    threshold = float(
        train_uncertainty.quantile(float(guard_config.get("uncertainty_quantile", 0.8)))
    )
    test_uncertainty = stage3.groupby("well_id")["pf_seed_std"].mean()
    row_uncertainty = stage3["well_id"].map(test_uncertainty).to_numpy(dtype=float)
    high_uncertainty = row_uncertainty > threshold
    stage3_correction = stage3["y_pred"].to_numpy(dtype=float) - stage3[
        "base_y_pred"
    ].to_numpy(dtype=float)
    cap = float(guard_config.get("correction_cap", 1.0))
    guarded_correction = stage3_correction.copy()
    guarded_correction[high_uncertainty] = np.clip(
        guarded_correction[high_uncertainty], -cap, cap
    )
    guarded = stage3["base_y_pred"].to_numpy(dtype=float) + guarded_correction

    trellis_parts: list[pd.DataFrame] = []
    for well_id, prediction in stage3.groupby("well_id", sort=False):
        path = path_map[str(well_id)]
        ordered = prediction.sort_values("row_index")
        correction, diagnostics = predict_trellis_correction(
            load_horizontal_well(path), load_typewell(path), ordered, trellis_config
        )
        trellis_parts.append(
            pd.DataFrame(
                {
                    "id": ordered["id"].to_numpy(),
                    "trellis_correction": correction,
                    **{name: value for name, value in diagnostics.items()},
                }
            )
        )
    trellis = pd.concat(trellis_parts, ignore_index=True)
    predictions = stage3.merge(trellis, on="id", how="left", validate="one_to_one")
    guarded_by_id = pd.Series(guarded, index=stage3["id"]).reindex(predictions["id"]).to_numpy()
    predictions["stage3_y_pred"] = predictions["y_pred"]
    predictions["guard_high_uncertainty"] = predictions["well_id"].map(test_uncertainty) > threshold
    predictions["guarded_y_pred"] = guarded_by_id
    predictions["y_pred"] = guarded_by_id + float(trellis_config.get("blend_weight", 0.1)) * predictions[
        "trellis_correction"
    ]

    sample_path = args.data_dir / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(f"Unexpected sample submission columns: {sample.columns.tolist()}")
    if sample["id"].duplicated().any() or predictions["id"].duplicated().any():
        raise ValueError("Submission or prediction IDs contain duplicates")
    sample_ids = set(sample["id"])
    prediction_ids = set(predictions["id"])
    if sample_ids != prediction_ids:
        raise ValueError(
            f"Submission ID mismatch: missing={len(sample_ids - prediction_ids)} "
            f"extra={len(prediction_ids - sample_ids)}"
        )
    submission = sample[["id"]].merge(
        predictions[["id", "y_pred"]], on="id", how="left", validate="one_to_one"
    )
    submission.rename(columns={"y_pred": "tvt"}, inplace=True)
    if not np.isfinite(submission["tvt"]).all():
        raise ValueError("Submission contains non-finite TVT predictions")
    submission.to_csv(output_dir / "submission.csv", index=False)
    stage3_submission = sample[["id"]].merge(
        predictions[["id", "stage3_y_pred"]], on="id", how="left", validate="one_to_one"
    )
    stage3_submission.rename(columns={"stage3_y_pred": "tvt"}, inplace=True)
    if not np.isfinite(stage3_submission["tvt"]).all():
        raise ValueError("Stage 3 submission contains non-finite TVT predictions")
    stage3_submission.to_csv(output_dir / "submission_stage3.csv", index=False)
    predictions.to_parquet(output_dir / "test_predictions.parquet", index=False)
    stage4_values = submission["tvt"].to_numpy(dtype=float)
    stage3_values = stage3_submission["tvt"].to_numpy(dtype=float)
    report = {
        "n_rows": len(submission),
        "n_wells": int(predictions["well_id"].nunique()),
        "id_order_matches_sample": submission["id"].equals(sample["id"]),
        "duplicate_ids": int(submission["id"].duplicated().sum()),
        "missing_tvt": int(submission["tvt"].isna().sum()),
        "tvt_min": float(submission["tvt"].min()),
        "tvt_max": float(submission["tvt"].max()),
        "stage3_stage4_prediction_rmse": float(
            np.sqrt(np.mean(np.square(stage3_values - stage4_values)))
        ),
        "stage3_stage4_prediction_correlation": float(
            np.corrcoef(stage3_values, stage4_values)[0, 1]
        ),
        "guard_uncertainty_threshold": threshold,
        "guarded_wells": sorted(
            test_uncertainty[test_uncertainty > threshold].index.astype(str).tolist()
        ),
        "per_well": predictions.groupby("well_id")["y_pred"].agg(["count", "min", "max"]).to_dict(
            orient="index"
        ),
    }
    write_json(output_dir / "submission_report.json", report)
    write_json(output_dir / "environment.json", environment_report())
    config["resolved"] = {
        "stage3_run": str(args.stage3_run.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(output_dir),
    }
    write_yaml(output_dir / "config.yaml", config)
    print(json.dumps(report, indent=2))
    print(f"primary Stage 4 submission: {output_dir / 'submission.csv'}")
    print(f"secondary Stage 3 submission: {output_dir / 'submission_stage3.csv'}")


if __name__ == "__main__":
    main()
