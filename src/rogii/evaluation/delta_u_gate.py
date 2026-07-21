from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from rogii.evaluation.metrics import evaluate_predictions, per_well_metrics


def absolute_tail_metrics(predictions: pd.DataFrame, fraction: float = 0.10) -> dict[str, float | int]:
    wells = per_well_metrics(predictions)
    count = max(1, math.ceil(len(wells) * float(fraction)))
    worst = wells.nlargest(count, "sse")
    total_sse = float(wells["sse"].sum())
    worst_sse = float(worst["sse"].sum())
    return {
        "tail_fraction": float(fraction),
        "tail_wells": int(count),
        "total_sse": total_sse,
        "worst_tail_sse": worst_sse,
        "worst_tail_sse_share": worst_sse / total_sse if total_sse else 0.0,
        "well_rmse_cvar": float(worst["rmse"].mean()),
        "well_rmse_p90": float(wells["rmse"].quantile(0.90)),
        "well_rmse_max": float(wells["rmse"].max()),
    }


def prediction_report(
    base: pd.DataFrame,
    prediction: np.ndarray,
    *,
    base_metrics: dict[str, Any] | None = None,
    base_tail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = base.copy()
    candidate["y_pred"] = np.asarray(prediction, dtype=float)
    candidate_metrics, _ = evaluate_predictions(candidate)
    if base_metrics is None:
        base_metrics, _ = evaluate_predictions(base)
    if base_tail is None:
        base_tail = absolute_tail_metrics(base)
    fold_deltas = {
        key: float(candidate_metrics[key] - base_metrics[key])
        for key in sorted(candidate_metrics)
        if key.startswith("fold_") and key.endswith("_rmse")
    }
    return {
        "pooled_rmse": float(candidate_metrics["pooled_rmse"]),
        "pooled_rmse_delta": float(
            candidate_metrics["pooled_rmse"] - base_metrics["pooled_rmse"]
        ),
        "fold_deltas": fold_deltas,
        "candidate_metrics": candidate_metrics,
        "base_metrics": base_metrics,
        "candidate_tail": absolute_tail_metrics(candidate),
        "base_tail": base_tail,
    }


def _eligible_on_selection(
    base: pd.DataFrame,
    prediction: np.ndarray,
    *,
    minimum_gain: float,
    inner_fold_tolerance: float,
    tail_tolerance: float,
    base_metrics: dict[str, Any] | None = None,
    base_tail: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    report = prediction_report(
        base,
        prediction,
        base_metrics=base_metrics,
        base_tail=base_tail,
    )
    tail = report["candidate_tail"]
    base_tail = report["base_tail"]
    eligible = bool(
        report["pooled_rmse_delta"] <= -float(minimum_gain)
        and max(report["fold_deltas"].values(), default=0.0) <= float(inner_fold_tolerance)
        and tail["worst_tail_sse"]
        <= base_tail["worst_tail_sse"] * (1.0 + float(tail_tolerance))
        and tail["well_rmse_cvar"]
        <= base_tail["well_rmse_cvar"] * (1.0 + float(tail_tolerance))
        and tail["well_rmse_p90"]
        <= base_tail["well_rmse_p90"] * (1.0 + float(tail_tolerance))
    )
    return eligible, report


def nested_select_predictions(
    base: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    folds = base["fold"].to_numpy(dtype=int)
    nested = base.copy()
    nested_prediction = base["y_pred"].to_numpy(dtype=float).copy()
    selections: list[dict[str, Any]] = []
    for outer_fold in sorted(np.unique(folds)):
        selection_mask = folds != int(outer_fold)
        outer_mask = ~selection_mask
        selection_base = base.loc[selection_mask].copy()
        selection_base_metrics, _ = evaluate_predictions(selection_base)
        selection_base_tail = absolute_tail_metrics(selection_base)
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for name, values in predictions.items():
            eligible, report = _eligible_on_selection(
                selection_base,
                np.asarray(values)[selection_mask],
                minimum_gain=float(config.get("minimum_selection_gain", 0.10)),
                inner_fold_tolerance=float(config.get("inner_fold_tolerance", 0.0)),
                tail_tolerance=float(config.get("tail_tolerance", 0.0)),
                base_metrics=selection_base_metrics,
                base_tail=selection_base_tail,
            )
            if eligible:
                ranked.append((float(report["pooled_rmse"]), name, report))
        if ranked:
            _, selected_name, selected_report = min(ranked, key=lambda item: (item[0], item[1]))
            nested_prediction[outer_mask] = np.asarray(predictions[selected_name])[outer_mask]
            selected_gain = float(-selected_report["pooled_rmse_delta"])
            worst_inner_delta = float(max(selected_report["fold_deltas"].values()))
        else:
            selected_name = None
            selected_gain = 0.0
            worst_inner_delta = None
        selections.append(
            {
                "fold": int(outer_fold),
                "eligible_specs": len(ranked),
                "selected_spec": selected_name,
                "selection_gain": selected_gain,
                "worst_inner_fold_delta": worst_inner_delta,
                "outer_rows": int(outer_mask.sum()),
            }
        )
    nested["y_pred"] = nested_prediction
    return nested, selections


def select_robust_inference_spec(
    family_reports: dict[str, dict[str, dict[str, Any]]],
    config: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    shared = set.intersection(
        *(set(reports) for reports in family_reports.values())
    ) if family_reports else set()
    rows: list[dict[str, Any]] = []
    for name in sorted(shared):
        eligible = True
        relative_rmse: list[float] = []
        worst_fold_delta = -float("inf")
        for family, reports in family_reports.items():
            report = reports[name]
            candidate_tail = report["candidate_tail"]
            base_tail = report["base_tail"]
            worst_fold_delta = max(
                worst_fold_delta,
                max(report["fold_deltas"].values(), default=0.0),
            )
            tail_limit = 1.0 + float(config.get("tail_tolerance", 0.0))
            eligible &= bool(
                report["pooled_rmse_delta"] < 0.0
                and max(report["fold_deltas"].values(), default=0.0)
                <= float(config.get("inference_fold_tolerance", 0.0))
                and candidate_tail["worst_tail_sse"]
                <= base_tail["worst_tail_sse"] * tail_limit
                and candidate_tail["well_rmse_cvar"]
                <= base_tail["well_rmse_cvar"] * tail_limit
                and candidate_tail["well_rmse_p90"]
                <= base_tail["well_rmse_p90"] * tail_limit
            )
            relative_rmse.append(
                float(report["pooled_rmse"] / report["base_metrics"]["pooled_rmse"])
            )
        rows.append(
            {
                "spec": name,
                "eligible": bool(eligible),
                "mean_relative_rmse": float(np.mean(relative_rmse)),
                "worst_fold_delta": float(worst_fold_delta),
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    selected = (
        min(eligible_rows, key=lambda row: (row["mean_relative_rmse"], row["spec"]))["spec"]
        if eligible_rows
        else None
    )
    return selected, rows
