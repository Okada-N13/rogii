from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True, nogil=True)
def _viterbi_offsets(
    emission: np.ndarray,
    max_jump: int,
    transition_penalty: float,
    zero_penalty: float,
    initial_penalty: float,
) -> np.ndarray:
    n_steps, n_states = emission.shape
    center = (n_states - 1) / 2.0
    previous = np.empty(n_states, dtype=np.float64)
    back = np.empty((n_steps, n_states), dtype=np.int16)
    for state in range(n_states):
        distance = state - center
        previous[state] = emission[0, state] + initial_penalty * distance * distance
        back[0, state] = -1
    for step in range(1, n_steps):
        current = np.empty(n_states, dtype=np.float64)
        for state in range(n_states):
            best_cost = 1e300
            best_previous = state
            lower = max(0, state - max_jump)
            upper = min(n_states - 1, state + max_jump)
            for source in range(lower, upper + 1):
                jump = state - source
                cost = previous[source] + transition_penalty * jump * jump
                if cost < best_cost:
                    best_cost = cost
                    best_previous = source
            distance = state - center
            current[state] = best_cost + emission[step, state] + zero_penalty * distance * distance
            back[step, state] = best_previous
        previous = current
    states = np.empty(n_steps, dtype=np.int16)
    states[-1] = int(np.argmin(previous))
    for step in range(n_steps - 1, 0, -1):
        states[step - 1] = back[step, states[step]]
    return states


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    left = window // 2
    right = window - left
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    result = np.empty(len(values), dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - left)
        stop = min(len(values), index + right)
        result[index] = (cumulative[stop] - cumulative[start]) / (stop - start)
    return result


def predict_trellis_correction(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    prediction: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    ordered = prediction.sort_values("row_index")
    rows = ordered["row_index"].to_numpy(dtype=np.int64)
    base = ordered["y_pred"].to_numpy(dtype=np.float64)
    full_gr_series = pd.to_numeric(horizontal["GR"], errors="coerce").interpolate(limit_direction="both")
    if full_gr_series.isna().all():
        raise ValueError(f"{ordered['well_id'].iloc[0]}: horizontal GR has no finite values")
    full_gr = full_gr_series.fillna(float(full_gr_series.mean())).to_numpy(dtype=np.float64)
    typewell_ordered = typewell.sort_values("TVT")
    typewell_gr_series = typewell_ordered["GR"].interpolate(limit_direction="both")
    if typewell_gr_series.isna().all():
        raise ValueError(f"{ordered['well_id'].iloc[0]}: typewell GR has no finite values")
    typewell_tvt = typewell_ordered["TVT"].to_numpy(dtype=np.float64)
    typewell_gr = typewell_gr_series.fillna(float(typewell_gr_series.mean())).to_numpy(dtype=np.float64)

    first_hidden = int(rows[0])
    known = horizontal.iloc[:first_hidden]
    valid = known["TVT_input"].notna() & known["GR"].notna()
    if int(valid.sum()) >= 20:
        known_expected = np.interp(
            known.loc[valid, "TVT_input"].to_numpy(dtype=float), typewell_tvt, typewell_gr
        )
        known_observed = known.loc[valid, "GR"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(known_expected)), known_expected])
        coefficients = np.linalg.lstsq(design, known_observed, rcond=None)[0]
        gain = float(np.clip(coefficients[1], 0.5, 1.5))
        offset = float(np.clip(np.median(known_observed - gain * known_expected), -50.0, 50.0))
        residual = known_observed - (offset + gain * known_expected)
        sigma = float(np.clip(np.median(np.abs(residual - np.median(residual))) * 1.4826, 8.0, 60.0))
    else:
        gain, offset, sigma = 1.0, 0.0, 30.0

    stride = int(config.get("tracking_stride", 4))
    tracked_positions = np.arange(0, len(ordered), stride, dtype=np.int64)
    if tracked_positions[-1] != len(ordered) - 1:
        tracked_positions = np.append(tracked_positions, len(ordered) - 1)
    tracked_rows = rows[tracked_positions]
    observed = full_gr[tracked_rows]
    tracked_base = base[tracked_positions]
    maximum_offset = float(config.get("maximum_offset", 40.0))
    offset_step = float(config.get("offset_step", 1.0))
    offsets = np.arange(-maximum_offset, maximum_offset + 0.5 * offset_step, offset_step)
    emission = np.empty((len(tracked_positions), len(offsets)), dtype=np.float64)
    window = int(config.get("evidence_window", 9))
    for state, candidate_offset in enumerate(offsets):
        expected = offset + gain * np.interp(
            tracked_base + candidate_offset,
            typewell_tvt,
            typewell_gr,
        )
        point_cost = np.square((observed - expected) / sigma)
        point_cost = np.minimum(point_cost, float(config.get("emission_clip", 16.0)))
        emission[:, state] = _rolling_mean(point_cost, window)

    states = _viterbi_offsets(
        emission,
        max_jump=int(config.get("max_jump_states", 2)),
        transition_penalty=float(config.get("transition_penalty", 0.3)),
        zero_penalty=float(config.get("zero_penalty", 0.0005)),
        initial_penalty=float(config.get("initial_penalty", 0.2)),
    )
    tracked_correction = offsets[states]
    target_md = ordered["MD"].to_numpy(dtype=float)
    tracked_md = target_md[tracked_positions]
    correction = np.interp(target_md, tracked_md, tracked_correction)
    diagnostics = {
        "trellis_gain": gain,
        "trellis_gr_offset": offset,
        "trellis_gr_sigma": sigma,
        "trellis_mean_abs_correction": float(np.mean(np.abs(correction))),
    }
    return correction, diagnostics
