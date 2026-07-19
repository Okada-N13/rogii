from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.data.loading import discover_horizontal_wells, load_horizontal_well, well_id_from_path
from rogii.evaluation.metrics import evaluate_predictions, paired_well_bootstrap
from rogii.models.spatial import (
    apply_spatial_correction,
    fit_well_residual_targets,
    make_spatial_blocks,
    spatial_knn_predictions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Stage 5 spatial residual correction")
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
    spatial_config = dict(config.get("spatial", {}))
    output_dir = args.artifact_dir.resolve() / args.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(args.base_run / "oof.parquet")
    if args.limit_wells is not None:
        selected = base["well_id"].drop_duplicates().iloc[: args.limit_wells]
        base = base[base["well_id"].isin(selected)].copy()
    targets = fit_well_residual_targets(base)
    first_rows = base.groupby("well_id")["row_index"].min().to_dict()
    coordinate_records: list[dict[str, float | str]] = []
    for path in discover_horizontal_wells(args.data_dir, "train"):
        well_id = well_id_from_path(path)
        if well_id not in first_rows:
            continue
        horizontal = load_horizontal_well(path)
        anchor = horizontal.iloc[int(first_rows[well_id]) - 1]
        coordinate_records.append(
            {"well_id": well_id, "x": float(anchor["X"]), "y": float(anchor["Y"])}
        )
    wells = targets.merge(pd.DataFrame.from_records(coordinate_records), on="well_id", validate="one_to_one")
    wells["spatial_fold"] = make_spatial_blocks(
        wells,
        n_blocks=int(spatial_config.get("n_spatial_blocks", 6)),
        seed=int(config.get("seed", 42)),
    )

    blend_weight = float(spatial_config.get("blend_weight", 0.1))
    correction_cap = float(spatial_config.get("correction_cap_ft", 10.0))
    seed = int(config.get("seed", 42))
    standard_wells = spatial_knn_predictions(wells, "fold", spatial_config, seed=seed)
    standard = apply_spatial_correction(
        base, standard_wells, blend_weight, correction_cap, "fold"
    )
    block_wells = spatial_knn_predictions(wells, "spatial_fold", spatial_config, seed=seed)
    block = apply_spatial_correction(
        base, block_wells, blend_weight, correction_cap, "spatial_fold"
    )
    shuffle_wells = spatial_knn_predictions(
        wells, "fold", spatial_config, shuffle_targets=True, seed=seed
    )
    shuffle = apply_spatial_correction(
        base, shuffle_wells, blend_weight, correction_cap, "fold"
    )
    base_standard = base.copy()
    base_block = base.drop(columns=["fold"]).merge(
        wells[["well_id", "spatial_fold"]], on="well_id", how="left", validate="many_to_one"
    )
    base_block["fold"] = base_block["spatial_fold"]

    standard_metrics, standard_well_metrics = evaluate_predictions(standard)
    block_metrics, _ = evaluate_predictions(block)
    shuffle_metrics, _ = evaluate_predictions(shuffle)
    base_metrics, _ = evaluate_predictions(base_standard)
    base_block_metrics, _ = evaluate_predictions(base_block)
    bootstrap = paired_well_bootstrap(standard, base_standard, seed=seed)
    standard_fold_deltas = {
        f"fold_{fold}": standard_metrics[f"fold_{fold}_rmse"]
        - base_metrics[f"fold_{fold}_rmse"]
        for fold in sorted(int(value) for value in standard["fold"].unique())
    }
    block_fold_deltas = {
        f"fold_{fold}": block_metrics[f"fold_{fold}_rmse"]
        - base_block_metrics[f"fold_{fold}_rmse"]
        for fold in sorted(int(value) for value in block["fold"].unique())
    }
    gates = {
        "standard_delta_at_most_minus_0_05": standard_metrics["pooled_rmse"]
        - base_metrics["pooled_rmse"]
        <= -0.05,
        "all_standard_folds_nonworse": max(standard_fold_deltas.values()) <= 0.0,
        "spatial_block_delta_at_most_minus_0_02": block_metrics["pooled_rmse"]
        - base_block_metrics["pooled_rmse"]
        <= -0.02,
        "all_spatial_blocks_nonworse": max(block_fold_deltas.values()) <= 0.0,
        "shuffle_does_not_improve": shuffle_metrics["pooled_rmse"]
        - base_metrics["pooled_rmse"]
        >= 0.0,
        "bootstrap_upper_below_zero": bootstrap["ci_97_5"] < 0.0,
        "worst_10pct_share_nonworse": standard_metrics["worst_10pct_sse_share"]
        <= base_metrics["worst_10pct_sse_share"],
    }
    summary = {
        "base_pooled_rmse": base_metrics["pooled_rmse"],
        "standard_pooled_rmse": standard_metrics["pooled_rmse"],
        "standard_delta": standard_metrics["pooled_rmse"] - base_metrics["pooled_rmse"],
        "spatial_block_pooled_rmse": block_metrics["pooled_rmse"],
        "spatial_block_delta": block_metrics["pooled_rmse"] - base_block_metrics["pooled_rmse"],
        "shuffle_pooled_rmse": shuffle_metrics["pooled_rmse"],
        "shuffle_delta": shuffle_metrics["pooled_rmse"] - base_metrics["pooled_rmse"],
        "promoted": all(gates.values()),
        "gates": gates,
        "standard_fold_deltas": standard_fold_deltas,
        "spatial_block_fold_deltas": block_fold_deltas,
        "bootstrap": bootstrap,
        "standard_metrics": standard_metrics,
        "spatial_block_metrics": block_metrics,
        "shuffle_metrics": shuffle_metrics,
    }
    standard.to_parquet(output_dir / "oof.parquet", index=False)
    block.to_parquet(output_dir / "spatial_block_oof.parquet", index=False)
    shuffle.to_parquet(output_dir / "shuffle_oof.parquet", index=False)
    standard_well_metrics.to_parquet(output_dir / "per_well_metrics.parquet", index=False)
    wells.to_parquet(output_dir / "spatial_wells.parquet", index=False)
    write_json(output_dir / "metrics.json", standard_metrics)
    write_json(output_dir / "spatial_audit.json", summary)
    write_json(output_dir / "environment.json", environment_report())
    config["resolved"] = {
        "base_run": str(args.base_run.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "run_id": args.run_id,
        "limit_wells": args.limit_wells,
    }
    write_yaml(output_dir / "config.yaml", config)
    print(
        pd.Series(summary).drop(
            labels=[
                "bootstrap",
                "gates",
                "standard_fold_deltas",
                "spatial_block_fold_deltas",
                "standard_metrics",
                "spatial_block_metrics",
                "shuffle_metrics",
            ]
        )
    )
    print("gates:", gates)
    print(f"run artifacts: {output_dir}")


if __name__ == "__main__":
    main()
