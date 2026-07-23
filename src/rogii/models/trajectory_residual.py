from __future__ import annotations

import numpy as np


COEFFICIENT_COLUMNS = ("target_residual_level", "target_residual_slope", "target_residual_curve")


def residual_basis(n_rows: int, ramp_rows: float = 96.0) -> np.ndarray:
    """Smooth three-term basis that preserves the visible-prefix boundary."""
    if int(n_rows) <= 0:
        return np.empty((0, len(COEFFICIENT_COLUMNS)), dtype=np.float64)
    step = np.arange(1, int(n_rows) + 1, dtype=np.float64)
    x = step / float(max(int(n_rows), 1))
    ramp = 1.0 - np.exp(-step / max(float(ramp_rows), 1.0))
    return np.column_stack((ramp, ramp * x, ramp * (2.0 * np.square(x) - x)))


def fit_residual_coefficients(
    base_prediction: np.ndarray,
    truth: np.ndarray,
    *,
    ramp_rows: float = 96.0,
    ridge: float = 1.0,
) -> np.ndarray:
    base = np.asarray(base_prediction, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if base.shape != target.shape or base.ndim != 1:
        raise ValueError("base_prediction and truth must be equal-length vectors")
    design = residual_basis(len(base), ramp_rows)
    finite = np.isfinite(base) & np.isfinite(target) & np.isfinite(design).all(axis=1)
    if int(finite.sum()) < design.shape[1]:
        return np.zeros(design.shape[1], dtype=np.float64)
    x = design[finite]
    y = target[finite] - base[finite]
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(ridge)
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def apply_residual_coefficients(
    base_prediction: np.ndarray,
    coefficients: np.ndarray,
    *,
    weight: float,
    cap_ft: float,
    ramp_rows: float = 96.0,
) -> np.ndarray:
    base = np.asarray(base_prediction, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if base.ndim != 1 or coefficients.shape != (len(COEFFICIENT_COLUMNS),):
        raise ValueError("invalid base vector or residual coefficient shape")
    raw = residual_basis(len(base), ramp_rows) @ coefficients
    correction = np.clip(float(weight) * raw, -float(cap_ft), float(cap_ft))
    output = base + correction
    if not np.isfinite(output).all():
        raise RuntimeError("trajectory residual correction produced non-finite predictions")
    return output
