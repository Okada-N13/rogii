from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.evaluation.metrics import evaluate_predictions, paired_well_bootstrap


def _fold_deltas(candidate_metrics: dict[str, float | int], base_metrics: dict[str, float | int]) -> dict[str, float]:
    names = sorted(name for name in candidate_metrics if name.startswith("fold_") and name.endswith("_rmse"))
    return {name: float(candidate_metrics[name]) - float(base_metrics[name]) for name in names}


def evaluate_candidate_gates(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    spatial_baseline: pd.DataFrame | None = None,
    spatial_candidate: pd.DataFrame | None = None,
    *,
    minimum_standard_gain: float = 0.05,
    minimum_spatial_gain: float = 0.02,
    minimum_improved_fold_fraction: float = 0.8,
    bootstrap_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    if len(baseline) != len(candidate):
        raise ValueError("Baseline and candidate row counts differ")
    if "id" in baseline and "id" in candidate:
        if not baseline["id"].reset_index(drop=True).equals(candidate["id"].reset_index(drop=True)):
            raise ValueError("Baseline and candidate IDs do not align")
    base_metrics, base_wells = evaluate_predictions(baseline)
    candidate_metrics, candidate_wells = evaluate_predictions(candidate)
    bootstrap = paired_well_bootstrap(
        candidate,
        baseline,
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    fold_deltas = _fold_deltas(candidate_metrics, base_metrics)
    improved_folds = sum(delta < 0.0 for delta in fold_deltas.values())
    required_folds = int(np.ceil(len(fold_deltas) * minimum_improved_fold_fraction))
    base_well_index = base_wells.set_index("well_id")
    candidate_well_index = candidate_wells.set_index("well_id").reindex(base_well_index.index)
    well_delta = candidate_well_index["rmse"] - base_well_index["rmse"]

    spatial_summary: dict[str, object] | None = None
    spatial_gate = True
    if spatial_baseline is not None or spatial_candidate is not None:
        if spatial_baseline is None or spatial_candidate is None:
            raise ValueError("Both spatial_baseline and spatial_candidate are required")
        spatial_base_metrics, _ = evaluate_predictions(spatial_baseline)
        spatial_candidate_metrics, _ = evaluate_predictions(spatial_candidate)
        spatial_fold_deltas = _fold_deltas(spatial_candidate_metrics, spatial_base_metrics)
        spatial_improved = sum(delta < 0.0 for delta in spatial_fold_deltas.values())
        spatial_required = int(np.ceil(len(spatial_fold_deltas) * minimum_improved_fold_fraction))
        spatial_delta = float(spatial_candidate_metrics["pooled_rmse"]) - float(
            spatial_base_metrics["pooled_rmse"]
        )
        spatial_gate = spatial_delta <= -minimum_spatial_gain and spatial_improved >= spatial_required
        spatial_summary = {
            "base_metrics": spatial_base_metrics,
            "candidate_metrics": spatial_candidate_metrics,
            "pooled_rmse_delta": spatial_delta,
            "fold_deltas": spatial_fold_deltas,
            "improved_folds": spatial_improved,
            "required_improved_folds": spatial_required,
        }

    pooled_delta = float(candidate_metrics["pooled_rmse"]) - float(base_metrics["pooled_rmse"])
    gates = {
        "standard_gain": pooled_delta <= -minimum_standard_gain,
        "fold_consistency": improved_folds >= required_folds,
        "bootstrap_upper_below_zero": float(bootstrap["ci_97_5"]) < 0.0,
        "well_p90_nonworse": float(candidate_metrics["well_rmse_p90"])
        <= float(base_metrics["well_rmse_p90"]) + 1e-12,
        "worst_10pct_share_nonworse": float(candidate_metrics["worst_10pct_sse_share"])
        <= float(base_metrics["worst_10pct_sse_share"]) + 1e-12,
        "spatial_gain": spatial_gate,
    }
    base_error = baseline["y_pred"].to_numpy(dtype=float) - baseline["y_true"].to_numpy(dtype=float)
    candidate_error = candidate["y_pred"].to_numpy(dtype=float) - candidate["y_true"].to_numpy(dtype=float)
    return {
        "promoted": all(gates.values()),
        "gates": gates,
        "base_metrics": base_metrics,
        "candidate_metrics": candidate_metrics,
        "pooled_rmse_delta": pooled_delta,
        "fold_deltas": fold_deltas,
        "improved_folds": improved_folds,
        "required_improved_folds": required_folds,
        "bootstrap": bootstrap,
        "improved_wells": int((well_delta < 0.0).sum()),
        "worsened_wells": int((well_delta > 0.0).sum()),
        "mean_well_rmse_delta": float(well_delta.mean()),
        "max_well_degradation": float(well_delta.max()),
        "error_correlation_with_baseline": float(np.corrcoef(base_error, candidate_error)[0, 1]),
        "spatial": spatial_summary,
    }
