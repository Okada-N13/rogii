from __future__ import annotations

import numpy as np


def optimal_gate(base: np.ndarray, full_candidate: np.ndarray, truth: np.ndarray) -> float:
    """Return the SSE-optimal interpolation coefficient between base and candidate."""
    error = np.asarray(base, float) - np.asarray(truth, float)
    movement = np.asarray(full_candidate, float) - np.asarray(base, float)
    denominator = float(np.dot(movement, movement))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(-np.dot(error, movement) / denominator, 0.0, 1.0))


def cut_gate_features(
    normalized: np.ndarray,
    raw_residual: np.ndarray,
    candidates: dict[str, np.ndarray],
    base_name: str,
    requested_fraction: float,
) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(normalized, float)
    residual = np.asarray(raw_residual, float)
    if values.ndim != 2 or len(values) != len(residual):
        raise ValueError("Cut gate feature inputs have incompatible shapes")
    vector = [
        float(requested_fraction),
        float(np.log1p(len(residual))),
        float(np.mean(residual)),
        float(np.std(residual)),
        float(np.mean(np.abs(residual))),
        float(np.quantile(np.abs(residual), 0.9)),
        float(residual[0]),
        float(residual[-1]),
        float(residual[-1] - residual[0]),
    ]
    names = [
        "requested_fraction",
        "log_suffix_rows",
        "residual_mean",
        "residual_std",
        "residual_mean_abs",
        "residual_abs_p90",
        "residual_first",
        "residual_last",
        "residual_endpoint_delta",
    ]
    for column in range(values.shape[1]):
        vector.extend([float(values[:, column].mean()), float(values[:, column].std())])
        names.extend([f"normalized_feature_{column}_mean", f"normalized_feature_{column}_std"])
    base = np.asarray(candidates[base_name], float)
    for candidate_name in sorted(candidates):
        if candidate_name == base_name:
            continue
        difference = np.asarray(candidates[candidate_name], float) - base
        vector.extend(
            [
                float(difference.mean()),
                float(difference.std()),
                float(np.mean(np.abs(difference))),
                float(difference[-1]),
            ]
        )
        names.extend(
            [
                f"{candidate_name}_minus_base_mean",
                f"{candidate_name}_minus_base_std",
                f"{candidate_name}_minus_base_mean_abs",
                f"{candidate_name}_minus_base_last",
            ]
        )
    output = np.asarray(vector, np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("Cut gate feature vector contains non-finite values")
    return output, names
