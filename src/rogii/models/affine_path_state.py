from __future__ import annotations

import numpy as np


def affine_path_library(
    steps: int,
    endpoint_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    endpoints = np.asarray(endpoint_values, float)
    start, end = np.meshgrid(endpoints, endpoints, indexing="ij")
    specifications = np.column_stack([start.ravel(), end.ravel()])
    progress = np.linspace(0.0, 1.0, int(steps), dtype=float)[:, None]
    corrections = (
        (1.0 - progress) * specifications[None, :, 0]
        + progress * specifications[None, :, 1]
    )
    return corrections, specifications


def aggregate_path_costs(
    row_state_costs: np.ndarray,
    offset_grid: np.ndarray,
    corrections: np.ndarray,
    invalid_cost: float = 2.999,
    missing_penalty: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    costs = np.asarray(row_state_costs, float)
    offsets = np.asarray(offset_grid, float)
    corrections = np.asarray(corrections, float)
    index = np.abs(corrections[:, :, None] - offsets[None, None, :]).argmin(axis=2)
    selected = np.take_along_axis(costs, index, axis=1)
    valid = np.isfinite(selected) & (selected < float(invalid_cost))
    valid_fraction = valid.mean(axis=0)
    masked = np.where(valid, selected, np.nan)
    with np.errstate(all="ignore"):
        score = np.nanmedian(masked, axis=0)
    score = np.nan_to_num(score, nan=float(invalid_cost), posinf=float(invalid_cost))
    score = score + float(missing_penalty) * (1.0 - valid_fraction)
    return score, valid_fraction


def decode_path_scores(
    scores: np.ndarray,
    corrections: np.ndarray,
    kind: str,
    temperature: float = 0.1,
) -> np.ndarray:
    scores = np.asarray(scores, float)
    corrections = np.asarray(corrections, float)
    if kind == "argmin":
        return corrections[:, int(np.argmin(scores))]
    if kind == "soft":
        shifted = -(scores - np.min(scores)) / max(float(temperature), 1e-6)
        weight = np.exp(np.clip(shifted, -80.0, 0.0))
        weight /= np.maximum(weight.sum(), 1e-12)
        return corrections @ weight
    raise ValueError(f"Unknown affine path decoder: {kind}")
