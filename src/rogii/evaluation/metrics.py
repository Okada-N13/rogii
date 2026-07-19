from __future__ import annotations

import math

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {"well_id", "MD", "y_true", "y_pred"}


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def per_well_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing columns: {missing}")
    if predictions.empty:
        raise ValueError("Predictions are empty")
    if predictions[["MD", "y_true", "y_pred"]].isna().any().any():
        raise ValueError("Evaluation columns contain missing values")

    records: list[dict[str, float | int | str]] = []
    for well_id, frame in predictions.groupby("well_id", sort=True):
        ordered = frame.sort_values("MD")
        y_true = ordered["y_true"].to_numpy(dtype=float)
        y_pred = ordered["y_pred"].to_numpy(dtype=float)
        error = y_pred - y_true
        if len(ordered) >= 2 and np.ptp(ordered["MD"].to_numpy(dtype=float)) > 0:
            md = ordered["MD"].to_numpy(dtype=float)
            true_slope = float(np.polyfit(md, y_true, 1)[0])
            pred_slope = float(np.polyfit(md, y_pred, 1)[0])
            slope_error = pred_slope - true_slope
        else:
            slope_error = float("nan")
        records.append(
            {
                "well_id": str(well_id),
                "n_rows": len(ordered),
                "sse": float(np.square(error).sum()),
                "rmse": _rmse(error),
                "bias": float(error.mean()),
                "mae": float(np.abs(error).mean()),
                "slope_error": slope_error,
            }
        )
    return pd.DataFrame.from_records(records)


def _worst_sse_share(wells: pd.DataFrame, fraction: float) -> float:
    count = max(1, math.ceil(len(wells) * fraction))
    total = float(wells["sse"].sum())
    if total == 0.0:
        return 0.0
    return float(wells.nlargest(count, "sse")["sse"].sum() / total)


def evaluate_predictions(predictions: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    wells = per_well_metrics(predictions)
    error = predictions["y_pred"].to_numpy(dtype=float) - predictions["y_true"].to_numpy(dtype=float)
    finite_slopes = wells["slope_error"].dropna().to_numpy(dtype=float)
    metrics: dict[str, float | int] = {
        "pooled_rmse": _rmse(error),
        "n_rows": len(predictions),
        "n_wells": len(wells),
        "well_rmse_median": float(wells["rmse"].median()),
        "well_rmse_p90": float(wells["rmse"].quantile(0.9)),
        "well_rmse_max": float(wells["rmse"].max()),
        "worst_5pct_sse_share": _worst_sse_share(wells, 0.05),
        "worst_10pct_sse_share": _worst_sse_share(wells, 0.10),
        "mean_bias_rmse": _rmse(wells["bias"].to_numpy(dtype=float)),
        "slope_error_rmse": _rmse(finite_slopes) if len(finite_slopes) else float("nan"),
    }
    if "fold" in predictions:
        for fold, frame in predictions.groupby("fold", sort=True):
            fold_error = frame["y_pred"].to_numpy(dtype=float) - frame["y_true"].to_numpy(dtype=float)
            metrics[f"fold_{int(fold)}_rmse"] = _rmse(fold_error)
    return metrics, wells


def paired_well_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    candidate_wells = per_well_metrics(candidate).set_index("well_id")
    baseline_wells = per_well_metrics(baseline).set_index("well_id")
    shared = candidate_wells.index.intersection(baseline_wells.index)
    if shared.empty:
        raise ValueError("Candidate and baseline have no shared wells")
    delta = candidate_wells.loc[shared, "rmse"].to_numpy() - baseline_wells.loc[shared, "rmse"].to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.choice(delta, size=(n_resamples, len(delta)), replace=True).mean(axis=1)
    return {
        "n_wells": len(shared),
        "n_resamples": n_resamples,
        "mean_well_rmse_delta": float(delta.mean()),
        "ci_2_5": float(np.quantile(samples, 0.025)),
        "ci_97_5": float(np.quantile(samples, 0.975)),
    }

