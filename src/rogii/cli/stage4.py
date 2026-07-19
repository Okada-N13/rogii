from __future__ import annotations

import argparse
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
from rogii.evaluation.metrics import evaluate_predictions
from rogii.models.trellis import predict_trellis_correction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Stage 4 tail guard and trellis blend")
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
    guard_config = dict(config.get("guard", {}))
    trellis_config = dict(config.get("trellis", {}))
    output_dir = args.artifact_dir.resolve() / args.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(args.base_run / "oof.parquet")
    if "base_y_pred" not in base or "pf_seed_std" not in base:
        raise ValueError("Stage 4 requires Stage 3 base_y_pred and PF diagnostics")
    if args.limit_wells is not None:
        selected = base["well_id"].drop_duplicates().iloc[: args.limit_wells]
        base = base[base["well_id"].isin(selected)].copy()
    paths = discover_horizontal_wells(args.data_dir, "train")
    path_map = {well_id_from_path(path): path for path in paths}

    corrections: list[pd.DataFrame] = []
    for index, (well_id, prediction) in enumerate(base.groupby("well_id", sort=False), start=1):
        path = path_map[str(well_id)]
        ordered = prediction.sort_values("row_index")
        correction, diagnostics = predict_trellis_correction(
            load_horizontal_well(path), load_typewell(path), ordered, trellis_config
        )
        part = pd.DataFrame(
            {
                "id": ordered["id"].to_numpy(),
                "trellis_correction": correction,
                **{name: value for name, value in diagnostics.items()},
            }
        )
        corrections.append(part)
        if index % 100 == 0:
            print(f"trellis {index}/{base['well_id'].nunique()} wells", flush=True)
    correction_frame = pd.concat(corrections, ignore_index=True)
    predictions = base.merge(correction_frame, on="id", how="left", validate="one_to_one")

    well_uncertainty = predictions.groupby("well_id")["pf_seed_std"].mean()
    quantile = float(guard_config.get("uncertainty_quantile", 1.0))
    threshold = float(well_uncertainty.quantile(quantile))
    uncertainty = predictions["well_id"].map(well_uncertainty).to_numpy(dtype=float)
    stage3_correction = predictions["y_pred"].to_numpy(dtype=float) - predictions[
        "base_y_pred"
    ].to_numpy(dtype=float)
    cap = float(guard_config.get("correction_cap", 1e9))
    high_uncertainty = uncertainty > threshold
    guarded_correction = stage3_correction.copy()
    guarded_correction[high_uncertainty] = np.clip(
        guarded_correction[high_uncertainty], -cap, cap
    )
    path_weight = float(trellis_config.get("blend_weight", 0.0))
    predictions["stage3_y_pred"] = predictions["y_pred"]
    predictions["guard_high_uncertainty"] = high_uncertainty
    predictions["guard_uncertainty_threshold"] = threshold
    predictions["guarded_y_pred"] = predictions["base_y_pred"] + guarded_correction
    predictions["y_pred"] = predictions["guarded_y_pred"] + path_weight * predictions[
        "trellis_correction"
    ]

    metrics, well_metrics = evaluate_predictions(predictions)
    predictions.to_parquet(output_dir / "oof.parquet", index=False)
    well_metrics.to_parquet(output_dir / "per_well_metrics.parquet", index=False)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "environment.json", environment_report())
    config["resolved"] = {
        "base_run": str(args.base_run.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "run_id": args.run_id,
        "limit_wells": args.limit_wells,
        "guard_uncertainty_threshold": threshold,
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
