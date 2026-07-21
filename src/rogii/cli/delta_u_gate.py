from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.data.loading import discover_horizontal_wells, load_horizontal_well, well_id_from_path
from rogii.evaluation.delta_u_gate import (
    absolute_tail_metrics,
    nested_select_predictions,
    prediction_report,
    select_robust_inference_spec,
)
from rogii.evaluation.metrics import evaluate_predictions, paired_well_bootstrap
from rogii.models.delta_u_surface import build_row_predictions


FOLD_FAMILIES = ("fold", "spatial_fold", "typewell_fold")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 11C robust delta-U weight/cap gate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage11-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser


def _specs(config: dict[str, Any]) -> list[dict[str, float | str]]:
    grid = dict(config.get("grid", {}))
    return [
        {
            "name": f"w{int(round(float(weight) * 100)):03d}_cap{int(round(float(cap))):02d}",
            "weight": float(weight),
            "cap": float(cap),
        }
        for weight, cap in product(
            grid.get("weights", [0.35, 0.50, 0.75, 1.00]),
            grid.get("correction_caps_ft", [30.0, 40.0, 50.0]),
        )
    ]


def _load_horizontal_by_well(data_dir: Path, well_ids: set[str]) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for path in discover_horizontal_wells(data_dir, "train"):
        well_id = well_id_from_path(path)
        if well_id in well_ids:
            outputs[well_id] = load_horizontal_well(path)
    missing = sorted(well_ids - set(outputs))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} Stage 11 wells, first={missing[:3]}")
    return outputs


def _family_candidates(
    coefficients: pd.DataFrame,
    horizontal_by_well: dict[str, pd.DataFrame],
    fold_column: str,
    stage11_evaluation: dict[str, Any],
    specs: list[dict[str, float | str]],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    base_config = dict(stage11_evaluation)
    base_config["correction_weight"] = 0.0
    base = build_row_predictions(coefficients, horizontal_by_well, fold_column, base_config)
    base["y_pred"] = base["base_y_pred"].to_numpy(dtype=float)
    base_metrics, _ = evaluate_predictions(base)
    base_tail = absolute_tail_metrics(base)
    predictions: dict[str, np.ndarray] = {}
    reports: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(specs, 1):
        candidate_config = dict(stage11_evaluation)
        candidate_config["correction_weight"] = float(spec["weight"])
        candidate_config["correction_cap_ft"] = float(spec["cap"])
        candidate = build_row_predictions(
            coefficients, horizontal_by_well, fold_column, candidate_config
        )
        if not candidate["id"].equals(base["id"]):
            raise RuntimeError(f"Candidate row order changed for {fold_column}/{spec['name']}")
        values = candidate["y_pred"].to_numpy(dtype=np.float32)
        predictions[str(spec["name"])] = values
        reports[str(spec["name"])] = prediction_report(
            base,
            values,
            base_metrics=base_metrics,
            base_tail=base_tail,
        )
        del candidate
        print(
            f"{fold_column} candidate {index}/{len(specs)} {spec['name']} "
            f"delta={reports[str(spec['name'])]['pooled_rmse_delta']:.6f}",
            flush=True,
        )
    return base, predictions, reports


def _compact_spec_rows(
    family: str,
    reports: dict[str, dict[str, Any]],
    spec_lookup: dict[str, dict[str, float | str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, report in reports.items():
        candidate_tail = report["candidate_tail"]
        base_tail = report["base_tail"]
        rows.append(
            {
                "family": family,
                "spec": name,
                "weight": float(spec_lookup[name]["weight"]),
                "cap": float(spec_lookup[name]["cap"]),
                "pooled_rmse": float(report["pooled_rmse"]),
                "pooled_rmse_delta": float(report["pooled_rmse_delta"]),
                "worst_fold_delta": float(max(report["fold_deltas"].values())),
                "worst_tail_sse_delta": float(
                    candidate_tail["worst_tail_sse"] - base_tail["worst_tail_sse"]
                ),
                "cvar_delta": float(
                    candidate_tail["well_rmse_cvar"] - base_tail["well_rmse_cvar"]
                ),
                "p90_delta": float(
                    candidate_tail["well_rmse_p90"] - base_tail["well_rmse_p90"]
                ),
                "worst_tail_share_delta": float(
                    candidate_tail["worst_tail_sse_share"]
                    - base_tail["worst_tail_sse_share"]
                ),
            }
        )
    return rows


def _cut_report(
    base: pd.DataFrame,
    nested: pd.DataFrame,
    coefficients: pd.DataFrame,
    fractions: list[float],
) -> list[dict[str, Any]]:
    cut_fraction = coefficients.set_index("cut_id")["cut_fraction"].to_dict()
    values = base["cut_id"].map(cut_fraction).to_numpy(dtype=float)
    targets = np.asarray(fractions, dtype=float)
    buckets = targets[np.argmin(np.abs(values[:, None] - targets[None, :]), axis=1)]
    rows: list[dict[str, Any]] = []
    for fraction in targets:
        mask = buckets == fraction
        candidate_metrics, _ = evaluate_predictions(nested.loc[mask])
        base_metrics, _ = evaluate_predictions(base.loc[mask])
        candidate_tail = absolute_tail_metrics(nested.loc[mask])
        base_tail = absolute_tail_metrics(base.loc[mask])
        rows.append(
            {
                "cut_fraction": float(fraction),
                "rows": int(mask.sum()),
                "base_rmse": float(base_metrics["pooled_rmse"]),
                "nested_rmse": float(candidate_metrics["pooled_rmse"]),
                "rmse_delta": float(
                    candidate_metrics["pooled_rmse"] - base_metrics["pooled_rmse"]
                ),
                "worst_tail_sse_delta": float(
                    candidate_tail["worst_tail_sse"] - base_tail["worst_tail_sse"]
                ),
                "cvar_delta": float(
                    candidate_tail["well_rmse_cvar"] - base_tail["well_rmse_cvar"]
                ),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    stage11_run = args.stage11_run.resolve()
    stage11_config = load_config(stage11_run / "config.yaml")
    stage11_evaluation = dict(stage11_config.get("evaluation", {}))
    selection_config = dict(config.get("selection", {}))
    validation_config = dict(config.get("validation", {}))
    output_dir = args.artifact_dir.resolve() / args.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    coefficients_by_family = {
        family: pd.read_parquet(stage11_run / f"{family}_coefficient_oof.parquet")
        for family in FOLD_FAMILIES
    }
    well_ids = set(coefficients_by_family["fold"]["well_id"].astype(str))
    horizontal_by_well = _load_horizontal_by_well(args.data_dir.resolve(), well_ids)
    specs = _specs(config)
    spec_lookup = {str(spec["name"]): spec for spec in specs}

    family_reports: dict[str, dict[str, dict[str, Any]]] = {}
    nested_reports: dict[str, dict[str, Any]] = {}
    nested_selections: dict[str, list[dict[str, Any]]] = {}
    compact_rows: list[dict[str, Any]] = []
    standard_base: pd.DataFrame | None = None
    standard_nested: pd.DataFrame | None = None
    for family in FOLD_FAMILIES:
        coefficients = coefficients_by_family[family]
        base, predictions, reports = _family_candidates(
            coefficients,
            horizontal_by_well,
            family,
            stage11_evaluation,
            specs,
        )
        nested, selections = nested_select_predictions(base, predictions, selection_config)
        nested_report = prediction_report(base, nested["y_pred"].to_numpy(dtype=float))
        family_reports[family] = reports
        nested_reports[family] = nested_report
        nested_selections[family] = selections
        compact_rows.extend(_compact_spec_rows(family, reports, spec_lookup))
        if family == "fold":
            standard_base = base
            standard_nested = nested
        del predictions, base, nested

    assert standard_base is not None and standard_nested is not None
    selected_spec, inference_report = select_robust_inference_spec(
        family_reports, selection_config
    )
    bootstrap = paired_well_bootstrap(
        standard_nested,
        standard_base,
        n_resamples=int(validation_config.get("bootstrap_resamples", 2000)),
        seed=int(config.get("seed", 42)),
    )
    cut_rows = _cut_report(
        standard_base,
        standard_nested,
        coefficients_by_family["fold"],
        [float(value) for value in validation_config.get("cut_fractions", [0.35, 0.50, 0.65, 0.80])],
    )
    standard_report = nested_reports["fold"]
    improved_folds = sum(
        delta < 0.0 for delta in standard_report["fold_deltas"].values()
    )
    required_folds = int(
        np.ceil(
            len(standard_report["fold_deltas"])
            * float(validation_config.get("minimum_improved_fold_fraction", 1.0))
        )
    )
    gates: dict[str, bool] = {
        "standard_nested_gain": standard_report["pooled_rmse_delta"]
        <= -float(validation_config.get("minimum_standard_gain", 0.50)),
        "standard_fold_consistency": improved_folds >= required_folds,
        "spatial_nested_gain": nested_reports["spatial_fold"]["pooled_rmse_delta"] < 0.0,
        "typewell_nested_gain": nested_reports["typewell_fold"]["pooled_rmse_delta"] < 0.0,
        "bootstrap_upper_below_zero": bootstrap["ci_97_5"] < 0.0,
        "standard_absolute_tail_nonworse": standard_report["candidate_tail"]["worst_tail_sse"]
        <= standard_report["base_tail"]["worst_tail_sse"],
        "standard_cvar_nonworse": standard_report["candidate_tail"]["well_rmse_cvar"]
        <= standard_report["base_tail"]["well_rmse_cvar"],
        "standard_p90_nonworse": standard_report["candidate_tail"]["well_rmse_p90"]
        <= standard_report["base_tail"]["well_rmse_p90"],
        "all_cut_fractions_improve": all(row["rmse_delta"] < 0.0 for row in cut_rows),
        "robust_inference_spec_available": selected_spec is not None,
    }
    summary = {
        "promoted_to_stage12": all(gates.values()),
        "experiment": "stage11c_delta_u_robust_gate",
        "source_stage11_run": str(stage11_run),
        "n_specs": len(specs),
        "standard_base_rmse": standard_report["base_metrics"]["pooled_rmse"],
        "standard_nested_rmse": standard_report["pooled_rmse"],
        "standard_nested_delta": standard_report["pooled_rmse_delta"],
        "spatial_nested_delta": nested_reports["spatial_fold"]["pooled_rmse_delta"],
        "typewell_nested_delta": nested_reports["typewell_fold"]["pooled_rmse_delta"],
        "improved_standard_folds": improved_folds,
        "required_standard_folds": required_folds,
        "bootstrap": bootstrap,
        "gates": gates,
        "selected_inference_spec": selected_spec,
        "selected_inference_parameters": spec_lookup.get(selected_spec) if selected_spec else None,
        "nested_selections": nested_selections,
        "nested_reports": nested_reports,
        "inference_spec_report": inference_report,
        "cut_report": cut_rows,
        "worst_tail_share_diagnostic_delta": float(
            standard_report["candidate_tail"]["worst_tail_sse_share"]
            - standard_report["base_tail"]["worst_tail_sse_share"]
        ),
        "next_step": (
            "Start Stage 12 raw-NCC and learned-emission alignment benchmark."
            if all(gates.values())
            else "Inspect nested selections and the failing absolute-tail/holdout gate before Stage 12."
        ),
    }
    fold_to_spec = {
        int(row["fold"]): row["selected_spec"] for row in nested_selections["fold"]
    }
    standard_nested["selected_spec"] = standard_nested["fold"].map(fold_to_spec)
    standard_nested.to_parquet(output_dir / "nested_oof.parquet", index=False)
    pd.DataFrame.from_records(compact_rows).to_parquet(
        output_dir / "spec_report.parquet", index=False
    )
    pd.DataFrame.from_records(cut_rows).to_parquet(
        output_dir / "cut_report.parquet", index=False
    )
    write_json(output_dir / "gate_summary.json", summary)
    write_json(output_dir / "environment.json", environment_report())
    config["resolved"] = {
        "stage11_run": str(stage11_run),
        "data_dir": str(args.data_dir.resolve()),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "run_id": args.run_id,
    }
    write_yaml(output_dir / "config.yaml", config)
    print(
        {
            "promoted_to_stage12": summary["promoted_to_stage12"],
            "base_rmse": summary["standard_base_rmse"],
            "nested_rmse": summary["standard_nested_rmse"],
            "rmse_delta": summary["standard_nested_delta"],
            "spatial_delta": summary["spatial_nested_delta"],
            "typewell_delta": summary["typewell_nested_delta"],
            "bootstrap_95pct": [bootstrap["ci_2_5"], bootstrap["ci_97_5"]],
            "improved_folds": f"{improved_folds}/{len(standard_report['fold_deltas'])}",
            "selected_inference_spec": selected_spec,
            "gates": gates,
        },
        flush=True,
    )
    print("standard selections:", nested_selections["fold"], flush=True)
    print("cut report:", cut_rows, flush=True)
    print(f"run artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
