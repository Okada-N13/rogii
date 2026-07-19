from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from rogii.data.schema import target_mask


@njit(cache=True)
def _interp_uniform(grid: np.ndarray, value: float, minimum: float, step: float) -> float:
    position = (value - minimum) / step
    index = int(position)
    if index < 0:
        return grid[0]
    last = len(grid) - 1
    if index >= last:
        return grid[last]
    fraction = position - index
    return grid[index] * (1.0 - fraction) + grid[index + 1] * fraction


@njit(cache=True, nogil=True)
def _run_pf_seeds(
    md: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    gr_grid: np.ndarray,
    grid_minimum: float,
    grid_step: float,
    gr_sigma: float,
    initial_surface: float,
    initial_rate: float,
    previous_md: float,
    n_particles: int,
    n_seeds: int,
    seed_base: int,
    momentum: float,
    rate_noise: float,
    position_noise: float,
    rough_position: float,
    rough_rate: float,
    resample_threshold: float,
    initial_spread: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = len(md)
    predictions = np.empty((n_seeds, n_steps), dtype=np.float64)
    log_likelihoods = np.empty(n_seeds, dtype=np.float64)
    grid_maximum = grid_minimum + (len(gr_grid) - 1) * grid_step
    for seed_index in range(n_seeds):
        np.random.seed(seed_base + seed_index)
        position = np.empty(n_particles, dtype=np.float64)
        rate = np.empty(n_particles, dtype=np.float64)
        weight = np.ones(n_particles, dtype=np.float64) / n_particles
        for particle in range(n_particles):
            position[particle] = initial_surface + initial_spread * np.random.randn()
            rate[particle] = initial_rate + 0.01 * np.random.randn()
        total_log_likelihood = 0.0
        last_md = previous_md
        for step_index in range(n_steps):
            md_step = md[step_index] - last_md
            if md_step < 1.0:
                md_step = 1.0
            noise_scale = np.sqrt(md_step)
            for particle in range(n_particles):
                rate[particle] = momentum * rate[particle] + rate_noise * noise_scale * np.random.randn()
                position[particle] += rate[particle] * md_step + position_noise * noise_scale * np.random.randn()
                particle_tvt = position[particle] - z[step_index]
                if particle_tvt < grid_minimum - 100.0:
                    particle_tvt = grid_minimum - 100.0
                if particle_tvt > grid_maximum + 100.0:
                    particle_tvt = grid_maximum + 100.0
                position[particle] = particle_tvt + z[step_index]

            average_likelihood = 0.0
            if not np.isnan(gr[step_index]):
                for particle in range(n_particles):
                    expected_gr = _interp_uniform(
                        gr_grid,
                        position[particle] - z[step_index],
                        grid_minimum,
                        grid_step,
                    )
                    residual = (gr[step_index] - expected_gr) / gr_sigma
                    squared = residual * residual
                    if squared > 600.0:
                        squared = 600.0
                    likelihood = np.exp(-0.5 * squared)
                    if likelihood < 1e-300:
                        likelihood = 1e-300
                    average_likelihood += weight[particle] * likelihood
                    weight[particle] *= likelihood
                if average_likelihood < 1e-300:
                    average_likelihood = 1e-300
                total_log_likelihood += np.log(average_likelihood)
                weight_sum = 0.0
                for particle in range(n_particles):
                    weight_sum += weight[particle]
                if weight_sum > 0.0:
                    for particle in range(n_particles):
                        weight[particle] /= weight_sum
                else:
                    for particle in range(n_particles):
                        weight[particle] = 1.0 / n_particles

            inverse_effective = 0.0
            for particle in range(n_particles):
                inverse_effective += weight[particle] * weight[particle]
            effective = 1.0 / inverse_effective
            if effective < resample_threshold * n_particles:
                cumulative = np.empty(n_particles, dtype=np.float64)
                cumulative_sum = 0.0
                for particle in range(n_particles):
                    cumulative_sum += weight[particle]
                    cumulative[particle] = cumulative_sum
                start = np.random.uniform(0.0, 1.0 / n_particles)
                new_position = np.empty(n_particles, dtype=np.float64)
                new_rate = np.empty(n_particles, dtype=np.float64)
                source = 0
                for particle in range(n_particles):
                    threshold = start + particle / n_particles
                    while source < n_particles - 1 and cumulative[source] < threshold:
                        source += 1
                    new_position[particle] = position[source] + rough_position * np.random.randn()
                    new_rate[particle] = rate[source] + rough_rate * np.random.randn()
                for particle in range(n_particles):
                    position[particle] = new_position[particle]
                    rate[particle] = new_rate[particle]
                    weight[particle] = 1.0 / n_particles

            estimate = 0.0
            for particle in range(n_particles):
                estimate += weight[particle] * (position[particle] - z[step_index])
            predictions[seed_index, step_index] = estimate
            last_md = md[step_index]
        log_likelihoods[seed_index] = total_log_likelihood
    return predictions, log_likelihoods


def _prepare_typewell(typewell: pd.DataFrame, grid_step: float) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    ordered = typewell[["TVT", "GR"]].copy().sort_values("TVT")
    ordered["GR"] = ordered["GR"].interpolate(limit_direction="both")
    if ordered["GR"].isna().all():
        raise ValueError("Typewell GR contains no finite values")
    ordered["GR"] = ordered["GR"].fillna(float(ordered["GR"].mean()))
    grouped = ordered.groupby("TVT", as_index=False)["GR"].mean()
    tvt = grouped["TVT"].to_numpy(dtype=np.float64)
    gr = grouped["GR"].to_numpy(dtype=np.float64)
    minimum = float(tvt.min())
    grid = np.arange(minimum, float(tvt.max()) + grid_step, grid_step, dtype=np.float64)
    gr_grid = np.interp(grid, tvt, gr).astype(np.float64)
    return gr_grid, minimum, float(grid_step), tvt, gr


def _well_seed(well_id: str, seed: int) -> int:
    try:
        value = int(well_id[:8], 16)
    except ValueError:
        value = sum((index + 1) * ord(character) for index, character in enumerate(well_id))
    return int((value + seed * 1_000_003) & 0x7FFFFFFF)


def predict_particle_filter(
    frame: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    hidden = target_mask(frame).to_numpy()
    first_hidden = int(np.flatnonzero(hidden)[0])
    known = frame.iloc[:first_hidden]
    target = frame.iloc[first_hidden:]
    anchor = known.iloc[-1]
    grid_step = float(config.get("grid_step", 0.2))
    gr_grid, grid_minimum, grid_step, typewell_tvt, typewell_gr = _prepare_typewell(typewell, grid_step)

    valid_prefix = known["GR"].notna() & known["TVT_input"].notna()
    if int(valid_prefix.sum()) >= 20:
        expected = np.interp(
            known.loc[valid_prefix, "TVT_input"].to_numpy(dtype=float),
            typewell_tvt,
            typewell_gr,
        )
        residual = known.loc[valid_prefix, "GR"].to_numpy(dtype=float) - expected
        gr_sigma = float(np.clip(np.nanstd(residual), 10.0, 60.0))
    else:
        gr_sigma = 30.0

    tail = known.tail(30)
    delta_tvt = np.diff(tail["TVT_input"].to_numpy(dtype=float))
    delta_z = np.diff(tail["Z"].to_numpy(dtype=float))
    delta_md = np.diff(tail["MD"].to_numpy(dtype=float))
    valid_delta = delta_md > 0
    initial_rate = (
        float(np.median((delta_tvt[valid_delta] + delta_z[valid_delta]) / delta_md[valid_delta]))
        if int(valid_delta.sum()) >= 3
        else 0.0
    )

    interpolated_gr = frame["GR"].interpolate(limit_direction="both").fillna(float(np.nanmean(typewell_gr)))
    tracking_stride = int(config.get("tracking_stride", 1))
    if tracking_stride < 1:
        raise ValueError("tracking_stride must be at least one")
    tracking_positions = np.arange(0, len(target), tracking_stride, dtype=np.int64)
    if tracking_positions[-1] != len(target) - 1:
        tracking_positions = np.append(tracking_positions, len(target) - 1)
    tracked = target.iloc[tracking_positions]
    tracked_gr = interpolated_gr.iloc[first_hidden:].iloc[tracking_positions].to_numpy(dtype=np.float64)

    well_id = str(frame["well_id"].iloc[0])
    likelihood_scale = float(config.get("likelihood_scale", 5.0))
    seed_batches = config.get("seed_batches", [int(config.get("seed", 42))])
    if not isinstance(seed_batches, list) or not seed_batches:
        raise ValueError("seed_batches must be a non-empty list")
    batch_predictions: list[np.ndarray] = []
    batch_variances: list[np.ndarray] = []
    batch_log_likelihoods: list[np.ndarray] = []
    for batch_seed in seed_batches:
        predictions, log_likelihoods = _run_pf_seeds(
            tracked["MD"].to_numpy(dtype=np.float64),
            tracked["Z"].to_numpy(dtype=np.float64),
            tracked_gr,
            gr_grid,
            grid_minimum,
            grid_step,
            gr_sigma,
            float(anchor["TVT_input"] + anchor["Z"]),
            initial_rate,
            float(anchor["MD"]),
            int(config.get("n_particles", 256)),
            int(config.get("n_seeds", 16)),
            _well_seed(well_id, int(batch_seed)),
            float(config.get("momentum", 0.998)),
            float(config.get("rate_noise", 0.002)),
            float(config.get("position_noise", 0.005)),
            float(config.get("rough_position", 0.1)),
            float(config.get("rough_rate", 0.001)),
            float(config.get("resample_threshold", 0.5)),
            float(config.get("initial_spread", 4.5)),
        )
        centered_likelihoods = log_likelihoods - float(log_likelihoods.max())
        seed_weights = np.exp(centered_likelihoods / likelihood_scale)
        seed_weights /= seed_weights.sum()
        batch_prediction = seed_weights @ predictions
        batch_predictions.append(batch_prediction)
        batch_variances.append(seed_weights @ np.square(predictions - batch_prediction[None, :]))
        batch_log_likelihoods.append(log_likelihoods)

    batch_matrix = np.vstack(batch_predictions)
    tracked_prediction = batch_matrix.mean(axis=0)
    tracked_variance = np.mean(
        np.vstack(batch_variances) + np.square(batch_matrix - tracked_prediction[None, :]),
        axis=0,
    )
    tracked_spread = np.sqrt(tracked_variance)
    all_log_likelihoods = np.concatenate(batch_log_likelihoods)

    target_md = target["MD"].to_numpy(dtype=float)
    tracked_md = tracked["MD"].to_numpy(dtype=float)
    prediction = np.interp(target_md, tracked_md, tracked_prediction)
    spread = np.interp(target_md, tracked_md, tracked_spread)
    anchor_tvt = float(anchor["TVT_input"])
    hold_weight = float(config.get("hold_weight", 0.0))
    prediction = (1.0 - hold_weight) * prediction + hold_weight * anchor_tvt

    result = pd.DataFrame(
        {
            "id": target["well_id"].astype(str) + "_" + target["row_index"].astype(str),
            "well_id": target["well_id"].astype(str).to_numpy(),
            "row_index": target["row_index"].to_numpy(dtype=np.int64),
            "MD": target_md,
            "Z": target["Z"].to_numpy(dtype=float),
            "model": "particle_filter",
            "anchor_value": anchor_tvt,
            "surface_anchor": float(anchor_tvt + anchor["Z"]),
            "pf_seed_std": spread,
            "pf_gr_sigma": gr_sigma,
            "pf_log_likelihood_spread": float(np.std(all_log_likelihoods)),
            "y_pred": prediction,
        }
    )
    if "TVT" in target:
        result["y_true"] = target["TVT"].to_numpy(dtype=float)
    return result
