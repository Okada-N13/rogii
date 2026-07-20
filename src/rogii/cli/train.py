from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config, resolve_artifact_dir, resolve_data_dir
from rogii.data.folds import make_group_folds
from rogii.data.loading import discover_horizontal_wells, load_horizontal_well, load_typewell
from rogii.data.schema import validate_horizontal_well
from rogii.evaluation.metrics import evaluate_predictions
from rogii.models.registry import predict_model


def _predict_path(path: Path, model_config: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame]:
    frame = load_horizontal_well(path)
    typewell = load_typewell(path)
    stats = validate_horizontal_well(frame, split="train")
    return stats.to_dict(), predict_model(frame, model_config, typewell)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a ROGII experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/baseline_anchor.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--limit-wells", type=int)
    return parser


def _default_run_id(experiment_name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{experiment_name}_{timestamp}"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    data_config = config.setdefault("data", {})
    cv_config = config.setdefault("cv", {})
    output_config = config.setdefault("output", {})
    data_dir = (args.data_dir or resolve_data_dir(data_config)).resolve()
    artifact_root = (args.artifact_dir or resolve_artifact_dir(output_config)).resolve()
    limit_wells = args.limit_wells if args.limit_wells is not None else data_config.get("limit_wells")
    run_id = args.run_id or _default_run_id(str(config.get("experiment_name", "experiment")))
    output_dir = artifact_root / run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = config.get("model", {})
    model_name = model_config.get("name")
    if not model_name:
        raise ValueError("model.name is required")

    paths = discover_horizontal_wells(data_dir, "train")
    if limit_wells is not None:
        paths = paths[: int(limit_wells)]
    stats_records: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    n_jobs = int(config.get("runtime", {}).get("n_jobs", 1))
    if n_jobs < 1:
        raise ValueError("runtime.n_jobs must be at least one")
    if n_jobs == 1:
        results = map(lambda path: _predict_path(path, model_config), paths)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=n_jobs, thread_name_prefix="rogii-well")
        results = executor.map(lambda path: _predict_path(path, model_config), paths)
    try:
        for index, (stats, prediction) in enumerate(results, start=1):
            stats_records.append(stats)
            prediction_frames.append(prediction)
            if index % 100 == 0:
                print(f"predicted {index}/{len(paths)} wells", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    stats_frame = pd.DataFrame.from_records(stats_records)
    n_splits = int(cv_config.get("n_splits", 5))
    seed = int(cv_config.get("seed", 42))
    folds = make_group_folds(stats_frame, n_splits=n_splits, seed=seed)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.merge(folds[["well_id", "fold"]], on="well_id", how="left", validate="many_to_one")
    if predictions["fold"].isna().any():
        raise RuntimeError("Some predictions have no fold assignment")

    metrics, well_metrics = evaluate_predictions(predictions)
    config["resolved"] = {
        "data_dir": str(data_dir),
        "artifact_dir": str(artifact_root),
        "run_id": run_id,
        "limit_wells": limit_wells,
        "n_jobs": n_jobs,
    }
    predictions.to_parquet(output_dir / "oof.parquet", index=False)
    well_metrics.to_parquet(output_dir / "per_well_metrics.parquet", index=False)
    folds.to_parquet(output_dir / "folds.parquet", index=False)
    stats_frame.to_parquet(output_dir / "well_stats.parquet", index=False)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "environment.json", environment_report())
    write_yaml(output_dir / "config.yaml", config)
    (output_dir / "run.log").write_text(
        f"model={model_name}\nwells={len(paths)}\nrows={len(predictions)}\npooled_rmse={metrics['pooled_rmse']:.8f}\n",
        encoding="utf-8",
    )
    print(f"pooled RMSE: {metrics['pooled_rmse']:.6f}")
    print(f"run artifacts: {output_dir}")


if __name__ == "__main__":
    main()
