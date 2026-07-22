from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def offset_grid(config: dict[str, Any]) -> np.ndarray:
    start = float(config.get("offset_min_ft", -60.0))
    stop = float(config.get("offset_max_ft", 60.0))
    step = float(config.get("offset_step_ft", 2.0))
    if step <= 0.0 or stop <= start:
        raise ValueError("Invalid NCC offset grid")
    return np.arange(start, stop + 0.5 * step, step, dtype=np.float32)


def surface_prediction(
    record: pd.Series | Any,
    horizontal: pd.DataFrame,
    *,
    weight: float,
    correction_cap_ft: float,
    slope_cap: float = 80.0,
    curvature_cap: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(int(record.cut_index), len(horizontal), dtype=np.int64)
    suffix = horizontal.iloc[positions]
    horizon_kft = (
        suffix["MD"].to_numpy(dtype=float) - float(record.anchor_md)
    ) / 1000.0
    slope = float(np.clip(record.pred_target_slope_correction, -slope_cap, slope_cap))
    curvature = float(np.clip(record.pred_target_curvature, -curvature_cap, curvature_cap))
    raw_correction = np.clip(
        slope * horizon_kft + curvature * np.square(horizon_kft),
        -float(correction_cap_ft),
        float(correction_cap_ft),
    )
    u = (
        float(record.anchor_u)
        + float(record.prefix_u_slope_per_kft) * horizon_kft
        + float(weight) * raw_correction
    )
    tvt = u - suffix["Z"].to_numpy(dtype=float)
    return positions, tvt


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError("NCC windows must be positive odd integers")
    before = window // 2
    after = window - before - 1
    padding = [(before, after)] + [(0, 0)] * (values.ndim - 1)
    padded = np.pad(values, padding, mode="constant", constant_values=0.0)
    cumulative = np.concatenate(
        [np.zeros((1, *padded.shape[1:]), dtype=np.float64), np.cumsum(padded, axis=0)],
        axis=0,
    )
    return cumulative[window:] - cumulative[:-window]


def rolling_ncc_cost(
    horizontal_gr: np.ndarray,
    expected_gr: np.ndarray,
    valid_expected: np.ndarray,
    window: int,
) -> np.ndarray:
    observed = np.asarray(horizontal_gr, dtype=float)
    expected = np.asarray(expected_gr, dtype=float)
    valid = np.asarray(valid_expected, dtype=bool) & np.isfinite(expected)
    observed_valid = np.isfinite(observed)
    joint = valid & observed_valid[:, None]
    x = np.where(joint, observed[:, None], 0.0)
    y = np.where(joint, expected, 0.0)
    count = _rolling_sum(joint.astype(float), window)
    sum_x = _rolling_sum(x, window)
    sum_y = _rolling_sum(y, window)
    sum_x2 = _rolling_sum(x * x, window)
    sum_y2 = _rolling_sum(y * y, window)
    sum_xy = _rolling_sum(x * y, window)
    safe_count = np.maximum(count, 1.0)
    covariance = sum_xy - sum_x * sum_y / safe_count
    variance_x = np.maximum(sum_x2 - sum_x * sum_x / safe_count, 0.0)
    variance_y = np.maximum(sum_y2 - sum_y * sum_y / safe_count, 0.0)
    denominator = np.sqrt(variance_x * variance_y)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 1e-8,
    )
    minimum = max(3, int(np.ceil(int(window) * 0.60)))
    cost = 1.0 - np.clip(correlation, -1.0, 1.0)
    cost[(count < minimum) | (denominator <= 1e-8)] = 3.0
    return cost.astype(np.float32)


def alignment_costs(
    record: pd.Series | Any,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
    *,
    weight: float,
    correction_cap_ft: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    positions, surface_tvt = surface_prediction(
        record,
        horizontal,
        weight=weight,
        correction_cap_ft=correction_cap_ft,
        slope_cap=float(config.get("slope_correction_cap", 80.0)),
        curvature_cap=float(config.get("curvature_cap", 30.0)),
    )
    stride = int(config.get("alignment_stride", 4))
    if stride < 1:
        raise ValueError("alignment_stride must be positive")
    take = np.arange(0, len(positions), stride, dtype=np.int64)
    positions = positions[take]
    surface_tvt = surface_tvt[take]
    observed_gr = pd.to_numeric(
        horizontal.iloc[positions]["GR"], errors="coerce"
    ).to_numpy(dtype=float)
    offsets = offset_grid(config)
    candidate_tvt = surface_tvt[:, None] + offsets[None, :]
    type_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(dtype=float)
    type_gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(type_tvt) & np.isfinite(type_gr)
    type_tvt = type_tvt[finite]
    type_gr = type_gr[finite]
    order = np.argsort(type_tvt)
    type_tvt = type_tvt[order]
    type_gr = type_gr[order]
    if len(type_tvt) < 5:
        raise ValueError(f"{record.well_id}: insufficient finite typewell GR")
    expected = np.interp(candidate_tvt.ravel(), type_tvt, type_gr).reshape(candidate_tvt.shape)
    valid_expected = (candidate_tvt >= type_tvt[0]) & (candidate_tvt <= type_tvt[-1])
    costs: dict[str, np.ndarray] = {}
    for window in [int(value) for value in config.get("windows", [5, 13, 25])]:
        costs[f"ncc_w{window}"] = rolling_ncc_cost(
            observed_gr, expected, valid_expected, window
        )
    mix_windows = [int(value) for value in config.get("mix_windows", [13, 25])]
    mix_weights = np.asarray(config.get("mix_weights", [0.4, 0.6]), dtype=float)
    if len(mix_windows) != len(mix_weights) or not np.isclose(mix_weights.sum(), 1.0):
        raise ValueError("mix_windows and mix_weights must have equal length and weights sum to one")
    mix = np.zeros_like(next(iter(costs.values())), dtype=np.float32)
    for window, mix_weight in zip(mix_windows, mix_weights):
        mix += float(mix_weight) * costs[f"ncc_w{window}"]
    costs["ncc_mix"] = mix
    return positions, surface_tvt, offsets, costs


def benchmark_cut(
    record: pd.Series | Any,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
    *,
    weight: float,
    correction_cap_ft: float,
    fold_column: str,
) -> pd.DataFrame:
    positions, surface_tvt, offsets, costs = alignment_costs(
        record,
        horizontal,
        typewell,
        config,
        weight=weight,
        correction_cap_ft=correction_cap_ft,
    )
    max_rows = int(config.get("max_eval_rows_per_cut", 512))
    if max_rows > 0 and len(positions) > max_rows:
        selected = np.unique(np.linspace(0, len(positions) - 1, max_rows).round().astype(int))
    else:
        selected = np.arange(len(positions), dtype=int)
    positions = positions[selected]
    surface_tvt = surface_tvt[selected]
    truth = horizontal.iloc[positions]["TVT"].to_numpy(dtype=float)
    true_offset = truth - surface_tvt
    true_state = np.argmin(np.abs(offsets[None, :] - true_offset[:, None]), axis=1)
    in_grid = (true_offset >= float(offsets[0])) & (true_offset <= float(offsets[-1]))
    oracle_offset = offsets[true_state]
    output = pd.DataFrame(
        {
            "id": str(record.cut_id) + "_" + horizontal.iloc[positions]["row_index"].astype(str),
            "well_id": str(record.well_id),
            "cut_id": str(record.cut_id),
            "cut_fraction": float(record.cut_fraction),
            "row_index": horizontal.iloc[positions]["row_index"].to_numpy(dtype=np.int64),
            "MD": horizontal.iloc[positions]["MD"].to_numpy(dtype=float),
            "y_true": truth,
            "surface_y_pred": surface_tvt,
            "true_offset": true_offset,
            "oracle_offset": oracle_offset,
            "oracle_y_pred": surface_tvt + oracle_offset,
            "offset_in_grid": in_grid,
            "fold": int(getattr(record, fold_column)),
        }
    )
    for name, full_cost in costs.items():
        cost = full_cost[selected]
        predicted_state = np.argmin(cost, axis=1)
        predicted_offset = offsets[predicted_state]
        true_cost = cost[np.arange(len(cost)), true_state]
        rank = 1 + np.sum(cost < true_cost[:, None], axis=1)
        emission_valid = in_grid & (true_cost < 2.999)
        output[f"{name}_offset"] = predicted_offset
        output[f"{name}_rank"] = rank.astype(np.int16)
        output[f"{name}_emission_valid"] = emission_valid
        output[f"{name}_top5"] = emission_valid & (rank <= 5)
        output[f"{name}_top10"] = emission_valid & (rank <= 10)
        output[f"{name}_y_pred"] = surface_tvt + predicted_offset
    return output
