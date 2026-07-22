from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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
from rogii.evaluation.delta_u_gate import absolute_tail_metrics
from rogii.evaluation.metrics import evaluate_predictions
from rogii.models.raw_ncc import alignment_costs, benchmark_cut, offset_grid


FOLD_FAMILIES = ("fold", "spatial_fold", "typewell_fold")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 12A raw multi-scale NCC benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage11-run", type=Path, required=True)
    parser.add_argument("--stage11c-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--limit-wells", type=int)
    return parser


def _load_wells(
    data_dir: Path,
    selected_wells: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    selected = set(selected_wells)
    horizontal: dict[str, pd.DataFrame] = {}
    typewell: dict[str, pd.DataFrame] = {}
    for path in discover_horizontal_wells(data_dir, "train"):
        well_id = well_id_from_path(path)
        if well_id in selected:
            horizontal[well_id] = load_horizontal_well(path)
            typewell[well_id] = load_typewell(path)
    missing = sorted(selected - set(horizontal))
    if missing:
        raise FileNotFoundError(f"Missing Stage 12 wells: {missing[:3]}")
    return horizontal, typewell


def _prediction_frame(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    output = frame[["id", "well_id", "cut_id", "cut_fraction", "MD", "y_true", "fold"]].copy()
    output["y_pred"] = frame[column].to_numpy(dtype=float)
    return output


def _variant_report(
    frame: pd.DataFrame,
    variant: str,
    n_states: int,
) -> dict[str, Any]:
    surface = _prediction_frame(frame, "surface_y_pred")
    aligned = _prediction_frame(frame, f"{variant}_y_pred")
    oracle = _prediction_frame(frame, "oracle_y_pred")
    surface_metrics, _ = evaluate_predictions(surface)
    aligned_metrics, _ = evaluate_predictions(aligned)
    oracle_metrics, _ = evaluate_predictions(oracle)
    valid = frame[f"{variant}_emission_valid"].to_numpy(dtype=bool)
    rank = frame.loc[valid, f"{variant}_rank"].to_numpy(dtype=float)
    true_offset = frame["true_offset"].to_numpy(dtype=float)
    predicted_offset = frame[f"{variant}_offset"].to_numpy(dtype=float)
    correlation = (
        float(np.corrcoef(true_offset, predicted_offset)[0, 1])
        if len(frame) > 1 and np.std(true_offset) > 0 and np.std(predicted_offset) > 0
        else 0.0
    )
    fold_top10 = {
        f"fold_{int(fold)}": float(group.loc[group[f"{variant}_emission_valid"], f"{variant}_top10"].mean())
        for fold, group in frame.groupby("fold", sort=True)
    }
    fold_aligned_deltas = {
        key: float(aligned_metrics[key] - surface_metrics[key])
        for key in aligned_metrics
        if key.startswith("fold_") and key.endswith("_rmse")
    }
    return {
        "variant": variant,
        "rows": int(len(frame)),
        "surface_rmse": float(surface_metrics["pooled_rmse"]),
        "aligned_rmse": float(aligned_metrics["pooled_rmse"]),
        "aligned_rmse_delta": float(
            aligned_metrics["pooled_rmse"] - surface_metrics["pooled_rmse"]
        ),
        "oracle_rmse": float(oracle_metrics["pooled_rmse"]),
        "oracle_rmse_delta": float(
            oracle_metrics["pooled_rmse"] - surface_metrics["pooled_rmse"]
        ),
        "offset_in_grid_fraction": float(frame["offset_in_grid"].mean()),
        "emission_valid_fraction": float(valid.mean()),
        "median_true_state_rank": float(np.median(rank)) if len(rank) else float(n_states),
        "mean_true_state_rank": float(np.mean(rank)) if len(rank) else float(n_states),
        "top5_recall": float(frame.loc[valid, f"{variant}_top5"].mean()) if valid.any() else 0.0,
        "top10_recall": float(frame.loc[valid, f"{variant}_top10"].mean()) if valid.any() else 0.0,
        "random_top5_recall": float(5.0 / n_states),
        "random_top10_recall": float(10.0 / n_states),
        "offset_correlation": correlation,
        "fold_top10_recall": fold_top10,
        "fold_aligned_rmse_deltas": fold_aligned_deltas,
        "surface_tail": absolute_tail_metrics(surface),
        "aligned_tail": absolute_tail_metrics(aligned),
    }


def _cut_report(frame: pd.DataFrame, variant: str, fractions: list[float]) -> list[dict[str, Any]]:
    values = frame["cut_fraction"].to_numpy(dtype=float)
    targets = np.asarray(fractions, dtype=float)
    buckets = targets[np.argmin(np.abs(values[:, None] - targets[None, :]), axis=1)]
    rows: list[dict[str, Any]] = []
    for fraction in targets:
        subset = frame.loc[buckets == fraction]
        report = _variant_report(subset, variant, n_states=61)
        rows.append(
            {
                "cut_fraction": float(fraction),
                "rows": int(len(subset)),
                "surface_rmse": report["surface_rmse"],
                "aligned_rmse_delta": report["aligned_rmse_delta"],
                "oracle_rmse": report["oracle_rmse"],
                "emission_valid_fraction": report["emission_valid_fraction"],
                "median_true_state_rank": report["median_true_state_rank"],
                "top5_recall": report["top5_recall"],
                "top10_recall": report["top10_recall"],
            }
        )
    return rows


def _invariance_check(
    coefficients: pd.DataFrame,
    horizontal: dict[str, pd.DataFrame],
    typewell: dict[str, pd.DataFrame],
    ncc_config: dict[str, Any],
    weight: float,
    cap: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for record in coefficients.head(min(6, len(coefficients))).itertuples(index=False):
        original = alignment_costs(
            record, horizontal[str(record.well_id)], typewell[str(record.well_id)], ncc_config,
            weight=weight, correction_cap_ft=cap,
        )
        changed = horizontal[str(record.well_id)].copy()
        changed.loc[changed.index[int(record.cut_index):], "TVT"] += 999.0
        perturbed = alignment_costs(
            record, changed, typewell[str(record.well_id)], ncc_config,
            weight=weight, correction_cap_ft=cap,
        )
        invariant = bool(
            np.array_equal(original[0], perturbed[0])
            and np.array_equal(original[1], perturbed[1])
            and all(
                np.array_equal(original[3][name], perturbed[3][name])
                for name in original[3]
            )
        )
        checks.append({"cut_id": str(record.cut_id), "invariant": invariant})
    return {"passed": bool(checks) and all(row["invariant"] for row in checks), "checks": checks}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    ncc_config = dict(config.get("ncc", {}))
    validation = dict(config.get("validation", {}))
    stage11_run = args.stage11_run.resolve()
    stage11c_run = args.stage11c_run.resolve()
    stage11c_summary = load_config(stage11c_run / "gate_summary.json")
    if not stage11c_summary.get("promoted_to_stage12", False):
        raise RuntimeError("Stage 11C did not authorize Stage 12")
    selected = dict(stage11c_summary["selected_inference_parameters"])
    weight = float(selected["weight"])
    cap = float(selected["cap"])
    required_name = str(validation.get("required_surface_spec", "w075_cap50"))
    if str(selected["name"]) != required_name:
        raise RuntimeError(
            f"Expected Stage 12 surface {required_name}, Stage 11C selected {selected['name']}"
        )
    output_dir = args.artifact_dir.resolve() / args.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    coefficients_by_family = {
        family: pd.read_parquet(stage11_run / f"{family}_coefficient_oof.parquet")
        for family in FOLD_FAMILIES
    }
    selected_wells = sorted(coefficients_by_family["fold"]["well_id"].astype(str).unique())
    if args.limit_wells is not None:
        selected_wells = selected_wells[: args.limit_wells]
        selected_set = set(selected_wells)
        coefficients_by_family = {
            family: frame[frame["well_id"].astype(str).isin(selected_set)].copy()
            for family, frame in coefficients_by_family.items()
        }
    horizontal, typewell = _load_wells(args.data_dir.resolve(), selected_wells)
    invariance = _invariance_check(
        coefficients_by_family["fold"], horizontal, typewell, ncc_config, weight, cap
    )
    if not invariance["passed"]:
        raise AssertionError(f"Stage 12 hidden-target invariance failed: {invariance}")

    family_reports: dict[str, dict[str, Any]] = {}
    standard_frame: pd.DataFrame | None = None
    variant_rows: list[dict[str, Any]] = []
    variants = [f"ncc_w{int(value)}" for value in ncc_config.get("windows", [5, 13, 25])] + ["ncc_mix"]
    for family in FOLD_FAMILIES:
        outputs: list[pd.DataFrame] = []
        coefficients = coefficients_by_family[family]
        for index, record in enumerate(coefficients.itertuples(index=False), 1):
            outputs.append(
                benchmark_cut(
                    record,
                    horizontal[str(record.well_id)],
                    typewell[str(record.well_id)],
                    ncc_config,
                    weight=weight,
                    correction_cap_ft=cap,
                    fold_column=family,
                )
            )
            if index % 100 == 0:
                print(f"{family}: benchmarked {index}/{len(coefficients)} cuts", flush=True)
        family_frame = pd.concat(outputs, ignore_index=True)
        reports = {
            variant: _variant_report(family_frame, variant, len(offset_grid(ncc_config)))
            for variant in variants
        }
        family_reports[family] = reports
        for variant, report in reports.items():
            variant_rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "surface_rmse": report["surface_rmse"],
                    "aligned_rmse": report["aligned_rmse"],
                    "aligned_rmse_delta": report["aligned_rmse_delta"],
                    "oracle_rmse": report["oracle_rmse"],
                    "emission_valid_fraction": report["emission_valid_fraction"],
                    "median_true_state_rank": report["median_true_state_rank"],
                    "top5_recall": report["top5_recall"],
                    "top10_recall": report["top10_recall"],
                    "offset_correlation": report["offset_correlation"],
                }
            )
        if family == "fold":
            standard_frame = family_frame
        else:
            del family_frame

    assert standard_frame is not None
    primary = str(validation.get("primary_variant", "ncc_mix"))
    primary_reports = {family: reports[primary] for family, reports in family_reports.items()}
    random_top10 = float(primary_reports["fold"]["random_top10_recall"])
    minimum_multiplier = float(validation.get("minimum_top10_random_multiplier", 1.25))
    fold_recall = primary_reports["fold"]["fold_top10_recall"]
    gates = {
        "hidden_target_invariance": invariance["passed"],
        "standard_valid_coverage": primary_reports["fold"]["emission_valid_fraction"]
        >= float(validation.get("minimum_emission_valid_fraction", 0.80)),
        "standard_top10_signal": primary_reports["fold"]["top10_recall"]
        >= random_top10 * minimum_multiplier,
        "standard_rank_signal": primary_reports["fold"]["median_true_state_rank"]
        <= float(validation.get("maximum_median_rank", 25.0)),
        "fold_recall_consistency": sum(value > random_top10 for value in fold_recall.values())
        >= int(validation.get("minimum_improved_recall_folds", 4)),
        "spatial_top10_signal": primary_reports["spatial_fold"]["top10_recall"]
        >= random_top10 * minimum_multiplier,
        "typewell_top10_signal": primary_reports["typewell_fold"]["top10_recall"]
        >= random_top10 * minimum_multiplier,
        "oracle_headroom": primary_reports["fold"]["oracle_rmse_delta"]
        <= -float(validation.get("minimum_oracle_gain", 2.0)),
    }
    cut_rows = _cut_report(
        standard_frame,
        primary,
        [float(value) for value in validation.get("cut_fractions", [0.35, 0.50, 0.65, 0.80])],
    )
    summary = {
        "promoted_to_learned_emission": all(gates.values()),
        "experiment": "stage12a_raw_ncc_benchmark",
        "surface_spec": selected,
        "offset_states": int(len(offset_grid(ncc_config))),
        "offset_min_ft": float(offset_grid(ncc_config)[0]),
        "offset_max_ft": float(offset_grid(ncc_config)[-1]),
        "primary_variant": primary,
        "n_wells": int(standard_frame["well_id"].nunique()),
        "n_cuts": int(standard_frame["cut_id"].nunique()),
        "standard_primary": primary_reports["fold"],
        "spatial_primary": primary_reports["spatial_fold"],
        "typewell_primary": primary_reports["typewell_fold"],
        "variant_reports": family_reports,
        "cut_report": cut_rows,
        "hidden_target_invariance": invariance,
        "gates": gates,
        "next_step": (
            "Train Stage 12B Siamese TCN emission on the fixed offset grid."
            if all(gates.values())
            else "Revise raw emission windows/grid before neural emission training."
        ),
    }
    standard_frame.to_parquet(output_dir / "standard_benchmark.parquet", index=False)
    pd.DataFrame.from_records(variant_rows).to_parquet(
        output_dir / "variant_report.parquet", index=False
    )
    pd.DataFrame.from_records(cut_rows).to_parquet(output_dir / "cut_report.parquet", index=False)
    write_json(output_dir / "benchmark_summary.json", summary)
    write_json(output_dir / "environment.json", environment_report())
    config["resolved"] = {
        "stage11_run": str(stage11_run),
        "stage11c_run": str(stage11c_run),
        "data_dir": str(args.data_dir.resolve()),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "run_id": args.run_id,
        "limit_wells": args.limit_wells,
    }
    write_yaml(output_dir / "config.yaml", config)
    print(
        {
            "promoted_to_learned_emission": summary["promoted_to_learned_emission"],
            "surface_spec": selected,
            "surface_rmse": primary_reports["fold"]["surface_rmse"],
            "raw_aligned_rmse": primary_reports["fold"]["aligned_rmse"],
            "raw_aligned_delta": primary_reports["fold"]["aligned_rmse_delta"],
            "oracle_rmse": primary_reports["fold"]["oracle_rmse"],
            "emission_valid_fraction": primary_reports["fold"]["emission_valid_fraction"],
            "median_true_state_rank": primary_reports["fold"]["median_true_state_rank"],
            "top5_recall": primary_reports["fold"]["top5_recall"],
            "top10_recall": primary_reports["fold"]["top10_recall"],
            "random_top10_recall": random_top10,
            "spatial_top10_recall": primary_reports["spatial_fold"]["top10_recall"],
            "typewell_top10_recall": primary_reports["typewell_fold"]["top10_recall"],
            "gates": gates,
            "cut_report": cut_rows,
            "next_step": summary["next_step"],
        },
        flush=True,
    )
    print(f"run artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
