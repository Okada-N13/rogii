from __future__ import annotations

import numpy as np


def resample_curve(values: np.ndarray, points: int) -> np.ndarray:
    source = np.asarray(values, float)
    if source.ndim != 1 or len(source) == 0:
        raise ValueError("Residual curve must be a non-empty vector")
    if len(source) == 1:
        return np.full(int(points), source[0], np.float32)
    source_axis = np.linspace(0.0, 1.0, len(source))
    target_axis = np.linspace(0.0, 1.0, int(points))
    return np.interp(target_axis, source_axis, source).astype(np.float32)


def restore_curve(grid_values: np.ndarray, rows: int) -> np.ndarray:
    values = np.asarray(grid_values, float)
    if values.ndim != 1 or len(values) == 0 or int(rows) <= 0:
        raise ValueError("Grid curve and output rows must be non-empty")
    if int(rows) == 1:
        return np.asarray([values[0]], float)
    grid_axis = np.linspace(0.0, 1.0, len(values))
    row_axis = np.linspace(0.0, 1.0, int(rows))
    return np.interp(row_axis, grid_axis, values)


def curve_descriptor(
    row_features: np.ndarray,
    candidates: dict[str, np.ndarray],
    base_name: str,
    requested_fraction: float,
) -> tuple[np.ndarray, list[str]]:
    features = np.asarray(row_features, float)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("Row features must be a non-empty matrix")
    vector = [float(requested_fraction), float(np.log1p(len(features)))]
    names = ["requested_fraction", "log_suffix_rows"]
    for column in range(features.shape[1]):
        values = features[:, column]
        vector.extend(
            [
                float(values.mean()),
                float(values.std()),
                float(values[0]),
                float(values[-1]),
            ]
        )
        names.extend(
            [
                f"row_feature_{column}_mean",
                f"row_feature_{column}_std",
                f"row_feature_{column}_first",
                f"row_feature_{column}_last",
            ]
        )
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
                float(difference[0]),
                float(difference[-1]),
            ]
        )
        names.extend(
            [
                f"{candidate_name}_minus_base_mean",
                f"{candidate_name}_minus_base_std",
                f"{candidate_name}_minus_base_mean_abs",
                f"{candidate_name}_minus_base_first",
                f"{candidate_name}_minus_base_last",
            ]
        )
    output = np.asarray(vector, np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("Functional residual descriptor contains non-finite values")
    return output, names


def equal_cut_optimal_alpha(
    bases: list[np.ndarray],
    full_candidates: list[np.ndarray],
    truths: list[np.ndarray],
    maximum: float,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for base, full, truth in zip(bases, full_candidates, truths, strict=True):
        error = np.asarray(base, float) - np.asarray(truth, float)
        movement = np.asarray(full, float) - np.asarray(base, float)
        rows = max(len(error), 1)
        numerator -= float(np.dot(error, movement)) / rows
        denominator += float(np.dot(movement, movement)) / rows
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, float(maximum)))
