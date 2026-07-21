from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.data.folds import assert_group_isolation, make_group_folds
from rogii.data.loading import discover_horizontal_wells, load_horizontal_well, load_typewell
from rogii.data.multicut import (
    TARGET_COLUMNS,
    build_cut_record,
    build_multicut_records,
    feature_columns,
    feature_schema_hash,
    make_cut_indices,
)
from rogii.evaluation.metrics import evaluate_predictions, paired_well_bootstrap
from rogii.models.delta_u_surface import build_row_predictions, crossfit_delta_u_coefficients
from rogii.models.spatial import make_spatial_blocks


TYPEWELL_CLUSTER_COLUMNS = [
    "typewell_gr_mean",
    "typewell_gr_std",
    "typewell_gr_q10",
    "typewell_gr_q50",
    "typewell_gr_q90",
    "typewell_tvt_min",
    "typewell_tvt_max",
    "typewell_tvt_span",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 11 multi-cut delta-U surface baseline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--limit-wells", type=int)
    return parser


def _well_assignments(records: pd.DataFrame, config: dict[str, Any], seed: int) -> pd.DataFrame:
    validation = dict(config.get("validation", {}))
    requested = int(validation.get("n_splits", 5))
    wells = records.groupby("well_id", as_index=False).first()
    n_splits = min(requested, len(wells))
    if n_splits < 2:
        raise ValueError("Stage 11 requires at least two wells")
    standard = make_group_folds(wells[["well_id"]], n_splits=n_splits, seed=seed)
    assert_group_isolation(standard)
    assignments = wells[["well_id", "anchor_x", "anchor_y", *TYPEWELL_CLUSTER_COLUMNS]].merge(
        standard[["well_id", "fold"]], on="well_id", validate="one_to_one"
    )
    spatial_blocks = min(int(validation.get("n_spatial_blocks", 6)), len(wells))
    typewell_blocks = min(int(validation.get("n_typewell_blocks", 5)), len(wells))
    if spatial_blocks < 2 or typewell_blocks < 2:
        raise ValueError("Spatial and typewell audits require at least two blocks")
    spatial_input = assignments.rename(columns={"anchor_x": "x", "anchor_y": "y"})
    assignments["spatial_fold"] = make_spatial_blocks(
        spatial_input, spatial_blocks, seed
    ).to_numpy()
    signatures = assignments[TYPEWELL_CLUSTER_COLUMNS].to_numpy(dtype=float)
    signatures = StandardScaler().fit_transform(signatures)
    assignments["typewell_fold"] = KMeans(
        n_clusters=typewell_blocks, random_state=seed, n_init=20
    ).fit_predict(signatures).astype(np.int16)
    return assignments


def _hidden_target_invariance(
    horizontal_by_well: dict[str, pd.DataFrame],
    typewell_by_well: dict[str, pd.DataFrame],
    multicut_config: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for well_id in sorted(horizontal_by_well)[: min(8, len(horizontal_by_well))]:
        horizontal = horizontal_by_well[well_id]
        cuts = make_cut_indices(
            len(horizontal),
            multicut_config.get("fractions", [0.35, 0.50, 0.65, 0.80]),
            int(multicut_config.get("min_prefix_rows", 64)),
            int(multicut_config.get("min_suffix_rows", 64)),
        )
        if not cuts:
            continue
        cut = cuts[len(cuts) // 2]
        record_options = {
            "prefix_window_ft": float(multicut_config.get("prefix_window_ft", 800.0)),
            "target_ridge": float(multicut_config.get("target_ridge", 1e-3)),
        }
        original = build_cut_record(
            horizontal, typewell_by_well[well_id], cut, **record_options
        )
        perturbed_frame = horizontal.copy()
        perturbed_frame.loc[perturbed_frame.index[cut:], "TVT"] += 999.0
        perturbed = build_cut_record(
            perturbed_frame, typewell_by_well[well_id], cut, **record_options
        )
        columns = feature_columns(pd.DataFrame([original]))
        left = np.asarray([original[column] for column in columns], dtype=float)
        right = np.asarray([perturbed[column] for column in columns], dtype=float)
        invariant = bool(np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True))
        target_changed = bool(
            any(not np.isclose(float(original[column]), float(perturbed[column])) for column in TARGET_COLUMNS)
        )
        checks.append(
            {
                "well_id": well_id,
                "cut_index": cut,
                "features_invariant": invariant,
                "target_changed": target_changed,
            }
        )
    passed = bool(checks) and all(
        row["features_invariant"] and row["target_changed"] for row in checks
    )
    return {"passed": passed, "checks": checks}


def _baseline_frame(candidate: pd.DataFrame) -> pd.DataFrame:
    baseline = candidate.copy()
    baseline["y_pred"] = baseline["base_y_pred"]
    return baseline


def _fold_deltas(candidate_metrics: dict[str, Any], base_metrics: dict[str, Any]) -> dict[str, float]:
    keys = sorted(key for key in candidate_metrics if key.startswith("fold_") and key.endswith("_rmse"))
    return {key: float(candidate_metrics[key] - base_metrics[key]) for key in keys}


def _evaluate_fold_family(
    records: pd.DataFrame,
    horizontal_by_well: dict[str, pd.DataFrame],
    fold_column: str,
    model_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, float], list[str]]:
    coefficients, columns = crossfit_delta_u_coefficients(
        records, fold_column, model_config, seed=seed
    )
    candidate = build_row_predictions(
        coefficients, horizontal_by_well, fold_column, evaluation_config
    )
    baseline = _baseline_frame(candidate)
    candidate_metrics, candidate_wells = evaluate_predictions(candidate)
    baseline_metrics, _ = evaluate_predictions(baseline)
    return (
        coefficients,
        candidate,
        candidate_metrics,
        baseline_metrics,
        _fold_deltas(candidate_metrics, baseline_metrics),
        columns,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    multicut_config = dict(config.get("multicut", {}))
    model_config = dict(config.get("model", {}))
    evaluation_config = dict(config.get("evaluation", {}))
    validation_config = dict(config.get("validation", {}))
    output_dir = args.artifact_dir.resolve() / args.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_horizontal_wells(args.data_dir, "train")
    if args.limit_wells is not None:
        paths = paths[: args.limit_wells]
    horizontal_by_well: dict[str, pd.DataFrame] = {}
    typewell_by_well: dict[str, pd.DataFrame] = {}
    record_frames: list[pd.DataFrame] = []
    for index, path in enumerate(paths, 1):
        horizontal = load_horizontal_well(path)
        typewell = load_typewell(path)
        well_id = str(horizontal["well_id"].iloc[0])
        cut_records = build_multicut_records(horizontal, typewell, multicut_config)
        if len(cut_records):
            horizontal_by_well[well_id] = horizontal
            typewell_by_well[well_id] = typewell
            record_frames.append(cut_records)
        if index % 100 == 0:
            print(f"built multi-cut records for {index}/{len(paths)} wells", flush=True)
    if not record_frames:
        raise RuntimeError("No eligible multi-cut training wells")
    records = pd.concat(record_frames, ignore_index=True)
    assignments = _well_assignments(records, config, seed)
    records = records.merge(
        assignments[["well_id", "fold", "spatial_fold", "typewell_fold"]],
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    invariance = _hidden_target_invariance(
        horizontal_by_well, typewell_by_well, multicut_config
    )
    if not invariance["passed"]:
        raise AssertionError(f"Hidden-target invariance failed: {invariance}")

    families: dict[str, dict[str, Any]] = {}
    standard_coefficients: pd.DataFrame | None = None
    standard_oof: pd.DataFrame | None = None
    standard_columns: list[str] = []
    standard_wells: pd.DataFrame | None = None
    for fold_column in ("fold", "spatial_fold", "typewell_fold"):
        print(f"cross-fitting {fold_column}", flush=True)
        coefficients, candidate, candidate_metrics, base_metrics, deltas, columns = _evaluate_fold_family(
            records,
            horizontal_by_well,
            fold_column,
            model_config,
            evaluation_config,
            seed,
        )
        families[fold_column] = {
            "candidate_metrics": candidate_metrics,
            "base_metrics": base_metrics,
            "pooled_rmse_delta": float(candidate_metrics["pooled_rmse"] - base_metrics["pooled_rmse"]),
            "fold_deltas": deltas,
        }
        coefficients.to_parquet(output_dir / f"{fold_column}_coefficient_oof.parquet", index=False)
        if fold_column == "fold":
            standard_coefficients = coefficients
            standard_oof = candidate
            standard_columns = columns
            _, standard_wells = evaluate_predictions(candidate)
        del candidate

    assert standard_coefficients is not None and standard_oof is not None and standard_wells is not None
    standard = families["fold"]
    spatial = families["spatial_fold"]
    typewell = families["typewell_fold"]
    standard_base = _baseline_frame(standard_oof)
    weight_report: list[dict[str, float]] = []
    for weight in [float(value) for value in evaluation_config.get("diagnostic_weights", [0.10, 0.20, 0.35, 0.50, 0.75, 1.00])]:
        diagnostic_config = dict(evaluation_config)
        diagnostic_config["correction_weight"] = weight
        diagnostic = build_row_predictions(
            standard_coefficients, horizontal_by_well, "fold", diagnostic_config
        )
        diagnostic_metrics, _ = evaluate_predictions(diagnostic)
        weight_report.append(
            {
                "weight": weight,
                "pooled_rmse": float(diagnostic_metrics["pooled_rmse"]),
                "pooled_rmse_delta": float(
                    diagnostic_metrics["pooled_rmse"]
                    - standard["base_metrics"]["pooled_rmse"]
                ),
                "well_rmse_p90": float(diagnostic_metrics["well_rmse_p90"]),
                "worst_10pct_sse_share": float(
                    diagnostic_metrics["worst_10pct_sse_share"]
                ),
            }
        )
        del diagnostic
    bootstrap = paired_well_bootstrap(
        standard_oof,
        standard_base,
        n_resamples=int(validation_config.get("bootstrap_resamples", 2000)),
        seed=seed,
    )
    improved_folds = sum(delta < 0.0 for delta in standard["fold_deltas"].values())
    required_folds = int(np.ceil(len(standard["fold_deltas"]) * float(validation_config.get("minimum_improved_fold_fraction", 0.8))))
    gates = {
        "hidden_target_invariance": invariance["passed"],
        "standard_gain": standard["pooled_rmse_delta"] <= -float(validation_config.get("minimum_standard_gain", 0.10)),
        "fold_consistency": improved_folds >= required_folds,
        "spatial_gain": spatial["pooled_rmse_delta"] < 0.0,
        "typewell_gain": typewell["pooled_rmse_delta"] < 0.0,
        "bootstrap_upper_below_zero": bootstrap["ci_97_5"] < 0.0,
        "well_p90_nonworse": standard["candidate_metrics"]["well_rmse_p90"] <= standard["base_metrics"]["well_rmse_p90"],
        "worst_10pct_share_nonworse": standard["candidate_metrics"]["worst_10pct_sse_share"] <= standard["base_metrics"]["worst_10pct_sse_share"],
    }
    summary = {
        "promoted_to_alignment_benchmark": all(gates.values()),
        "experiment": "stage11_multicut_delta_u_surface",
        "n_wells": int(records["well_id"].nunique()),
        "n_cuts": int(len(records)),
        "cuts_per_well": records.groupby("well_id").size().value_counts().sort_index().to_dict(),
        "feature_count": len(standard_columns),
        "feature_schema_hash": feature_schema_hash(standard_columns),
        "sampled_evaluation_rows": int(len(standard_oof)),
        "standard": standard,
        "spatial": spatial,
        "typewell": typewell,
        "improved_folds": improved_folds,
        "required_folds": required_folds,
        "bootstrap": bootstrap,
        "fixed_correction_weight": float(evaluation_config.get("correction_weight", 0.35)),
        "diagnostic_weight_report": weight_report,
        "gates": gates,
        "hidden_target_invariance": invariance,
        "next_step": (
            "Run the fixed raw-NCC alignment benchmark, then learned emission Stage 12."
            if all(gates.values())
            else "Diagnose delta-U coefficient errors before starting learned emission training."
        ),
    }
    records.to_parquet(output_dir / "multicut_records.parquet", index=False)
    assignments.to_parquet(output_dir / "well_folds.parquet", index=False)
    standard_oof.to_parquet(output_dir / "oof.parquet", index=False)
    standard_wells.to_parquet(output_dir / "per_well_metrics.parquet", index=False)
    write_json(output_dir / "gate_summary.json", summary)
    write_json(output_dir / "environment.json", environment_report())
    config["resolved"] = {
        "data_dir": str(args.data_dir.resolve()),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "run_id": args.run_id,
        "limit_wells": args.limit_wells,
    }
    write_yaml(output_dir / "config.yaml", config)
    print(
        {
            "promoted_to_alignment_benchmark": summary["promoted_to_alignment_benchmark"],
            "base_rmse": standard["base_metrics"]["pooled_rmse"],
            "candidate_rmse": standard["candidate_metrics"]["pooled_rmse"],
            "rmse_delta": standard["pooled_rmse_delta"],
            "spatial_delta": spatial["pooled_rmse_delta"],
            "typewell_delta": typewell["pooled_rmse_delta"],
            "improved_folds": f"{improved_folds}/{len(standard['fold_deltas'])}",
            "bootstrap_95pct": [bootstrap["ci_2_5"], bootstrap["ci_97_5"]],
            "gates": gates,
        },
        flush=True,
    )
    print(f"run artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
