from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from rogii.models.trellis import _viterbi_offsets


@njit(cache=True, nogil=True)
def _shape_emission(
    observed: np.ndarray,
    expected: np.ndarray,
    tracked: np.ndarray,
    half_windows: np.ndarray,
    scale_weights: np.ndarray,
    sigma: float,
    amplitude_weight: float,
) -> np.ndarray:
    n_states, n_rows = expected.shape
    emission = np.empty((len(tracked), n_states), dtype=np.float64)
    for tracked_index in range(len(tracked)):
        center = tracked[tracked_index]
        for state in range(n_states):
            cost = 0.0
            for scale_index in range(len(half_windows)):
                half = half_windows[scale_index]
                start = max(0, center - half)
                stop = min(n_rows, center + half + 1)
                count = stop - start
                observed_mean = 0.0
                expected_mean = 0.0
                for row in range(start, stop):
                    observed_mean += observed[row]
                    expected_mean += expected[state, row]
                observed_mean /= count
                expected_mean /= count
                covariance = 0.0
                observed_square = 0.0
                expected_square = 0.0
                for row in range(start, stop):
                    left = observed[row] - observed_mean
                    right = expected[state, row] - expected_mean
                    covariance += left * right
                    observed_square += left * left
                    expected_square += right * right
                denominator = np.sqrt(observed_square * expected_square)
                correlation = covariance / denominator if denominator > 1e-12 else 0.0
                if correlation < -1.0:
                    correlation = -1.0
                if correlation > 1.0:
                    correlation = 1.0
                cost += scale_weights[scale_index] * (1.0 - correlation)
            residual = (observed[center] - expected[state, center]) / sigma
            amplitude = residual * residual
            if amplitude > 16.0:
                amplitude = 16.0
            emission[tracked_index, state] = cost + amplitude_weight * amplitude
    return emission


def _prefix_shape_correlation(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> float:
    known = horizontal[horizontal["TVT_input"].notna() & horizontal["GR"].notna()]
    if len(known) < 30:
        return 0.0
    observed = known["GR"].to_numpy(dtype=float)
    expected = np.interp(known["TVT_input"].to_numpy(dtype=float), typewell_tvt, typewell_gr)
    observed = observed - pd.Series(observed).rolling(31, center=True, min_periods=1).mean().to_numpy()
    expected = expected - pd.Series(expected).rolling(31, center=True, min_periods=1).mean().to_numpy()
    denominator = float(np.sqrt(np.sum(observed**2) * np.sum(expected**2)))
    return float(np.sum(observed * expected) / denominator) if denominator > 1e-12 else 0.0


def predict_multiscale_alignment(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    prediction: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    ordered = prediction.sort_values("row_index")
    rows = ordered["row_index"].to_numpy(dtype=int)
    base = ordered["base_y_pred"].to_numpy(dtype=float)
    gr_series = pd.to_numeric(horizontal["GR"], errors="coerce").interpolate(limit_direction="both")
    if gr_series.isna().all():
        return np.zeros(len(ordered)), {"status": 0.0, "prefix_shape_corr": 0.0}
    gr_series = gr_series.fillna(float(gr_series.mean()))
    observed = gr_series.iloc[rows].to_numpy(dtype=float)

    tw = typewell[["TVT", "GR"]].copy().sort_values("TVT")
    tw["GR"] = tw["GR"].interpolate(limit_direction="both")
    tw = tw.dropna(subset=["TVT", "GR"]).groupby("TVT", as_index=False)["GR"].mean()
    typewell_tvt = tw["TVT"].to_numpy(dtype=float)
    typewell_gr = tw["GR"].to_numpy(dtype=float)
    if len(typewell_tvt) < 10:
        return np.zeros(len(ordered)), {"status": 0.0, "prefix_shape_corr": 0.0}

    known = horizontal[horizontal["TVT_input"].notna() & horizontal["GR"].notna()]
    if len(known) >= 20:
        reference = np.interp(known["TVT_input"].to_numpy(dtype=float), typewell_tvt, typewell_gr)
        actual = known["GR"].to_numpy(dtype=float)
        design = np.column_stack([reference, np.ones(len(reference))])
        gain, offset = np.linalg.lstsq(design, actual, rcond=None)[0]
        gain = float(np.clip(gain, 0.5, 1.5))
        offset = float(np.clip(np.median(actual - gain * reference), -50.0, 50.0))
        residual = actual - (gain * reference + offset)
        sigma = float(np.clip(1.4826 * np.median(np.abs(residual - np.median(residual))), 8.0, 60.0))
    else:
        gain, offset, sigma = 1.0, 0.0, 30.0

    maximum_offset = float(config.get("maximum_offset", 20.0))
    offset_step = float(config.get("offset_step", 1.0))
    offsets = np.arange(-maximum_offset, maximum_offset + 0.5 * offset_step, offset_step)
    expected = np.vstack(
        [offset + gain * np.interp(base + candidate_offset, typewell_tvt, typewell_gr) for candidate_offset in offsets]
    )
    stride = int(config.get("tracking_stride", 8))
    tracked = np.arange(0, len(base), stride, dtype=np.int64)
    if tracked[-1] != len(base) - 1:
        tracked = np.append(tracked, len(base) - 1)
    half_windows = np.asarray(config.get("half_windows", [4, 12, 30]), dtype=np.int64)
    scale_weights = np.asarray(config.get("scale_weights", [0.2, 0.4, 0.4]), dtype=float)
    scale_weights /= scale_weights.sum()
    emission = _shape_emission(
        observed,
        expected,
        tracked,
        half_windows,
        scale_weights,
        sigma,
        float(config.get("amplitude_weight", 0.05)),
    )
    states = _viterbi_offsets(
        emission,
        int(config.get("max_jump_states", 2)),
        float(config.get("transition_penalty", 0.08)),
        float(config.get("zero_penalty", 0.0008)),
        float(config.get("initial_penalty", 0.1)),
    )
    tracked_correction = offsets[states]
    correction = np.interp(np.arange(len(base)), tracked, tracked_correction)
    center = int(np.argmin(np.abs(offsets)))
    zero_cost = float(emission[:, center].sum())
    path_emission_cost = float(emission[np.arange(len(tracked)), states].sum())
    path_total_cost = path_emission_cost
    zero_penalty = float(config.get("zero_penalty", 0.0008))
    initial_penalty = float(config.get("initial_penalty", 0.1))
    state_center = 0.5 * (len(offsets) - 1)
    path_total_cost += initial_penalty * float(states[0] - state_center) ** 2
    path_total_cost += zero_penalty * float(np.sum(np.square(states[1:] - state_center)))
    for index in range(1, len(states)):
        jump = float(states[index] - states[index - 1])
        path_total_cost += float(config.get("transition_penalty", 0.08)) * jump * jump
    relative_gain = (zero_cost - path_total_cost) / max(len(tracked), 1)
    prefix_correlation = _prefix_shape_correlation(horizontal, typewell_tvt, typewell_gr)
    active = (
        relative_gain >= float(config.get("minimum_cost_gain", 0.0))
        and prefix_correlation >= float(config.get("minimum_prefix_correlation", -1.0))
    )
    if not active:
        correction[:] = 0.0
    return correction.astype(float), {
        "status": 1.0,
        "active": float(active),
        "prefix_shape_corr": prefix_correlation,
        "cost_gain_per_step": relative_gain,
        "mean_abs_correction": float(np.mean(np.abs(correction))),
        "max_abs_correction": float(np.max(np.abs(correction))),
        "gr_gain": gain,
        "gr_offset": offset,
        "gr_sigma": sigma,
    }
