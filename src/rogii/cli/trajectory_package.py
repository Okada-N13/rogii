from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.trajectory_residual import _cut_feature_record, _model
from rogii.config import load_config
from rogii.models.trajectory_residual import COEFFICIENT_COLUMNS, apply_residual_coefficients


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 19B all-data trajectory model bundle")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage19a-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--stage17b-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _train_bundle(
    records: pd.DataFrame,
    feature_columns: list[str],
    model_config: dict[str, Any],
    seeds: list[int],
    model_dir: Path,
) -> list[dict[str, Any]]:
    model_dir.mkdir(parents=True, exist_ok=True)
    x = records[feature_columns].replace([np.inf, -np.inf], np.nan)
    sample_weight = np.sqrt(records["suffix_rows"].to_numpy(float))
    report = []
    for seed in seeds:
        for index, target in enumerate(COEFFICIENT_COLUMNS):
            model = _model(model_config, int(seed) + index)
            model.fit(x, records[target], sample_weight=sample_weight)
            path = model_dir / f"seed_{int(seed)}__{target}.pkl"
            with path.open("wb") as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            prediction = model.predict(x)
            report.append({
                "seed": int(seed), "target": target, "file": f"models/{path.name}",
                "training_rmse": float(np.sqrt(np.mean(np.square(prediction - records[target].to_numpy(float))))),
                "bytes": path.stat().st_size, "sha256": _sha256(path),
            })
    return report


def _load_models(package_dir: Path, model_report: list[dict[str, Any]]) -> dict[str, list[Any]]:
    output = {target: [] for target in COEFFICIENT_COLUMNS}
    for item in model_report:
        with (package_dir / str(item["file"])).open("rb") as handle:
            output[str(item["target"])].append(pickle.load(handle))
    return output


def _predict_coefficients(models: dict[str, list[Any]], features: pd.DataFrame) -> np.ndarray:
    values = []
    for target in COEFFICIENT_COLUMNS:
        predictions = np.stack([model.predict(features) for model in models[target]])
        values.append(np.mean(predictions, axis=0))
    output = np.column_stack(values)
    if not np.isfinite(output).all():
        raise RuntimeError("Stage 19B coefficient ensemble produced non-finite values")
    return output


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    stage19a, stage17a, stage17b = args.stage19a_run.resolve(), args.stage17a_run.resolve(), args.stage17b_run.resolve()
    stage19_summary = json.loads((stage19a / "summary.json").read_text(encoding="utf-8"))
    if not stage19_summary.get("promoted_to_stage19b"):
        raise AssertionError("Stage 19A was not promoted")
    if not stage19_summary.get("hidden_target_invariance"):
        raise AssertionError("Stage 19A hidden-target invariance did not pass")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    package = output / "stage19b_trajectory_bundle"
    package.mkdir(parents=True, exist_ok=True)

    records = pd.read_parquet(stage19a / "cut_features.parquet")
    feature_columns = [str(value) for value in stage19_summary["feature_columns"]]
    missing = sorted(set(feature_columns) - set(records.columns))
    if missing:
        raise AssertionError(f"Stage 19A records are missing package features: {missing}")
    seeds = [int(value) for value in config.get("ensemble_seeds", [42, 52, 62, 72, 82])]
    model_report = _train_bundle(
        records, feature_columns, dict(config.get("model", {})), seeds, package / "models"
    )
    expected_models = len(seeds) * len(COEFFICIENT_COLUMNS)
    if len(model_report) != expected_models:
        raise AssertionError("Stage 19B model ensemble is incomplete")
    models = _load_models(package, model_report)

    benchmark = dict(config.get("benchmark", {}))
    target_fraction = float(benchmark.get("requested_fraction", 0.26))
    selected = (
        records.assign(_distance=(records["requested_fraction"] - target_fraction).abs())
        .sort_values(["well_id", "_distance", "cut_index"], kind="stable")
        .groupby("well_id", sort=True).head(1).drop(columns="_distance")
        .sort_values("well_id", kind="stable").reset_index(drop=True)
    )
    cut_ids = selected["cut_id"].astype(str).tolist()
    cut_report = pd.read_parquet(stage17b / "cut_report.parquet")
    cut_report["cut_id"] = cut_report["cut_id"].astype(str)
    cut_report = cut_report[cut_report["cut_id"].isin(cut_ids)].copy()
    eligible = cut_report.loc[cut_report["replay_eligible"], "cut_id"].astype(str).tolist()
    uncovered = cut_report.loc[~cut_report["replay_eligible"], "cut_id"].astype(str).tolist()
    prediction_columns = ["cut_id", "row_index", "y_pred"]
    parts = []
    if eligible:
        parts.append(pd.read_parquet(
            stage17a / "replay_predictions.parquet",
            columns=prediction_columns, filters=[("cut_id", "in", eligible)],
        ))
    if uncovered:
        parts.append(pd.read_parquet(
            stage17b / "selector_predictions.parquet",
            columns=prediction_columns, filters=[("cut_id", "in", uncovered)],
        ))
    base = pd.concat(parts, ignore_index=True)
    base["cut_id"] = base["cut_id"].astype(str)
    base_by_cut = base.groupby("cut_id", sort=False)
    train_dir = args.data_dir.resolve() / "train"

    @lru_cache(maxsize=64)
    def load_well(well_id: str) -> pd.DataFrame:
        frame = pd.read_csv(train_dir / f"{well_id}__horizontal_well.csv")
        frame["well_id"] = str(well_id)
        frame["row_index"] = np.arange(len(frame), dtype=np.int64)
        return frame

    @lru_cache(maxsize=64)
    def load_typewell(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train_dir / f"{well_id}__typewell.csv")

    feature_config = dict(config.get("features", {}))
    stored = records.set_index("cut_id")
    benchmark_rows = []
    started = time.perf_counter()
    for cut in selected.itertuples(index=False):
        well_id, cut_id, cut_index = str(cut.well_id), str(cut.cut_id), int(cut.cut_index)
        base_prediction = base_by_cut.get_group(cut_id).sort_values("row_index", kind="stable")["y_pred"].to_numpy(float)
        feature = _cut_feature_record(load_well(well_id), load_typewell(well_id), cut_index, base_prediction, feature_config)
        feature["requested_fraction"] = float(cut.requested_fraction)
        current = pd.DataFrame([{column: feature[column] for column in feature_columns}]).replace([np.inf, -np.inf], np.nan)
        reference = stored.loc[cut_id, feature_columns].to_numpy(float)
        recomputed = current.iloc[0].to_numpy(float)
        finite = np.isfinite(reference) & np.isfinite(recomputed)
        parity = bool(np.allclose(reference, recomputed, rtol=0.0, atol=1e-10, equal_nan=True))
        max_difference = float(np.max(np.abs(reference[finite] - recomputed[finite]))) if finite.any() else 0.0
        coefficients = _predict_coefficients(models, current)[0]
        profile = dict(stage19_summary["profile"])
        prediction = apply_residual_coefficients(
            base_prediction, coefficients, weight=float(profile["weight"]),
            cap_ft=float(profile["cap_ft"]), ramp_rows=float(profile["ramp_rows"]),
        )
        benchmark_rows.append({
            "well_id": well_id, "cut_id": cut_id, "cut_index": cut_index,
            "feature_parity": parity, "feature_max_abs_difference": max_difference,
            "prediction_rows": len(prediction), "prediction_finite": bool(np.isfinite(prediction).all()),
            **{f"pred_{target}": float(value) for target, value in zip(COEFFICIENT_COLUMNS, coefficients, strict=True)},
        })
    elapsed = float(time.perf_counter() - started)
    benchmark_frame = pd.DataFrame.from_records(benchmark_rows)
    benchmark_frame.to_parquet(output / "runtime_benchmark.parquet", index=False)
    estimated_wells = int(benchmark.get("estimated_hidden_wells", 200))
    estimated_seconds = elapsed / max(len(benchmark_frame), 1) * estimated_wells
    maximum_seconds = float(benchmark.get("maximum_estimated_seconds", 600.0))

    manifest = {
        "stage19b_trajectory_bundle": True, "package_version": 1,
        "source_stage19a_feature_schema_hash": stage19_summary["feature_schema_hash"],
        "feature_columns": feature_columns, "coefficient_columns": list(COEFFICIENT_COLUMNS),
        "ensemble_seeds": seeds, "features": feature_config,
        "profile": stage19_summary["profile"], "models": model_report,
        "runtime_contract": {
            "donor_search": False, "rowwise_neural_model": False,
            "models": expected_models, "outputs_per_well": len(COEFFICIENT_COLUMNS),
        },
    }
    write_json(package / "manifest.json", manifest)
    manifest_sha = _sha256(package / "manifest.json")
    archive = output / "stage19b_trajectory_bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(package))
    gates = {
        "stage19a_promoted": True,
        "hidden_target_invariance": True,
        "model_ensemble_complete": len(model_report) == expected_models,
        "feature_recompute_parity": bool(benchmark_frame["feature_parity"].all()),
        "finite_coefficients_and_predictions": bool(benchmark_frame["prediction_finite"].all()),
        "hidden_runtime_under_10_minutes": estimated_seconds <= maximum_seconds,
    }
    summary = {
        "stage19b_complete": True, "promoted_to_stage19c": bool(all(gates.values())),
        "training_cuts": len(records), "training_wells": int(records["well_id"].nunique()),
        "feature_count": len(feature_columns), "ensemble_seeds": seeds,
        "model_count": len(model_report), "model_report": model_report,
        "benchmark": {
            "wells": len(benchmark_frame), "elapsed_seconds": elapsed,
            "seconds_per_well": elapsed / max(len(benchmark_frame), 1),
            "estimated_hidden_wells": estimated_wells,
            "estimated_hidden_seconds": estimated_seconds,
            "maximum_estimated_seconds": maximum_seconds,
            "maximum_feature_difference": float(benchmark_frame["feature_max_abs_difference"].max()),
        },
        "package_manifest_sha256": manifest_sha, "zip": str(archive),
        "gates": gates,
        "next_step": (
            "Build Stage 19C Internet-OFF inference and integrate it after the frozen 6.589 base."
            if all(gates.values()) else
            "Fix package parity or runtime before integrating with the Kaggle notebook."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage19a_run": str(stage19a), "stage17a_run": str(stage17a),
        "stage17b_run": str(stage17b), "data_dir": str(args.data_dir.resolve()), "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
