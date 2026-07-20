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


def _robust_gr_calibration(
    observed: np.ndarray,
    reference: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, float, bool]:
    """Fit observed ~= gain * reference + offset using known prefix rows only."""
    minimum_points = int(config.get("minimum_points", 40))
    valid = np.isfinite(observed) & np.isfinite(reference)
    if int(valid.sum()) < minimum_points:
        return 1.0, 0.0, False
    x = reference[valid].astype(np.float64)
    y = observed[valid].astype(np.float64)
    design = np.column_stack([x, np.ones_like(x)])
    weights = np.ones_like(x)
    coefficients = np.array([1.0, 0.0], dtype=np.float64)
    for _ in range(int(config.get("robust_iterations", 6))):
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_target = y * np.sqrt(weights)
        coefficients = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)[0]
        residual = y - design @ coefficients
        median = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - median)))
        if not np.isfinite(scale) or scale < 1e-6:
            break
        cutoff = float(config.get("huber_delta", 1.5)) * scale
        absolute = np.abs(residual - median)
        weights = np.minimum(1.0, cutoff / np.maximum(absolute, 1e-12))

    gain = float(coefficients[0])
    offset = float(coefficients[1])
    gain_min = float(config.get("gain_min", 0.5))
    gain_max = float(config.get("gain_max", 2.0))
    offset_limit = float(config.get("offset_limit", 100.0))
    if not (np.isfinite(gain) and np.isfinite(offset) and gain_min <= gain <= gain_max):
        return 1.0, 0.0, False
    offset = float(np.clip(offset, -offset_limit, offset_limit))
    raw_rmse = float(np.sqrt(np.mean(np.square(y - x))))
    calibrated_rmse = float(np.sqrt(np.mean(np.square((y - offset) / gain - x))))
    required_ratio = float(config.get("required_rmse_ratio", 0.98))
    if calibrated_rmse > required_ratio * raw_rmse:
        return 1.0, 0.0, False
    return gain, offset, True


def _weighted_two_mode_hedge(
    levels: np.ndarray,
    weights: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, float, float]:
    """Return a direction-free midpoint hedge for a genuinely bimodal PF ensemble."""
    valid = np.isfinite(levels) & np.isfinite(weights) & (weights > 0.0)
    levels = levels[valid].astype(np.float64)
    weights = weights[valid].astype(np.float64)
    if len(levels) < 4 or float(weights.sum()) <= 0.0:
        return 0.0, 0.0, 0.0
    weights /= weights.sum()
    centers = np.quantile(levels, [0.10, 0.90]).astype(np.float64)
    for _ in range(20):
        assignment = np.abs(levels - centers[0]) > np.abs(levels - centers[1])
        updated = centers.copy()
        masses = np.empty(2, dtype=np.float64)
        for mode in range(2):
            mask = assignment == bool(mode)
            masses[mode] = float(weights[mask].sum())
            if masses[mode] > 0.0:
                updated[mode] = float(np.sum(weights[mask] * levels[mask]) / masses[mode])
        if np.allclose(updated, centers, rtol=0.0, atol=1e-8):
            centers = updated
            break
        centers = updated
    order = np.argsort(centers)
    centers = centers[order]
    masses = masses[order]
    separation = float(centers[1] - centers[0])
    minor_mass = float(masses.min())
    if (
        minor_mass < float(config.get("minimum_mass", 0.22))
        or separation < float(config.get("minimum_separation", 4.0))
        or separation > float(config.get("maximum_separation", 40.0))
    ):
        return 0.0, separation, minor_mass
    weighted_center = float(np.sum(weights * levels))
    midpoint = float(0.5 * (centers[0] + centers[1]))
    shift = float(config.get("alpha", 1.4)) * (midpoint - weighted_center)
    cap = float(config.get("cap", 4.0))
    return float(np.clip(shift, -cap, cap)), separation, minor_mass


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
    gr_gain = 1.0
    gr_offset = 0.0
    gr_calibrated = False
    calibration_config = config.get("heel_calibration", {})
    if int(valid_prefix.sum()) >= 20:
        expected = np.interp(
            known.loc[valid_prefix, "TVT_input"].to_numpy(dtype=float),
            typewell_tvt,
            typewell_gr,
        )
        observed_prefix = known.loc[valid_prefix, "GR"].to_numpy(dtype=float)
        if bool(calibration_config.get("enabled", False)):
            gr_gain, gr_offset, gr_calibrated = _robust_gr_calibration(
                observed_prefix,
                expected,
                calibration_config,
            )
        calibrated_prefix = (observed_prefix - gr_offset) / gr_gain
        residual = calibrated_prefix - expected
        residual_median = float(np.nanmedian(residual))
        robust_sigma = 1.4826 * float(np.nanmedian(np.abs(residual - residual_median)))
        if not np.isfinite(robust_sigma) or robust_sigma <= 0.0:
            robust_sigma = float(np.nanstd(residual))
        gr_sigma = float(np.clip(robust_sigma, 10.0, 60.0))
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

    calibrated_gr = (frame["GR"] - gr_offset) / gr_gain
    interpolated_gr = calibrated_gr.interpolate(limit_direction="both").fillna(float(np.nanmean(typewell_gr)))
    tracking_stride = int(config.get("tracking_stride", 1))
    if tracking_stride < 1:
        raise ValueError("tracking_stride must be at least one")
    tracking_positions = np.arange(0, len(target), tracking_stride, dtype=np.int64)
    if tracking_positions[-1] != len(target) - 1:
        tracking_positions = np.append(tracking_positions, len(target) - 1)
    tracked = target.iloc[tracking_positions]
    tracked_gr = interpolated_gr.iloc[first_hidden:].iloc[tracking_positions].to_numpy(dtype=np.float64)

    well_id = str(frame["well_id"].iloc[0])
    likelihood_scales_raw = config.get("likelihood_scales", [float(config.get("likelihood_scale", 5.0))])
    if not isinstance(likelihood_scales_raw, list) or not likelihood_scales_raw:
        raise ValueError("likelihood_scales must be a non-empty list")
    likelihood_scales = np.asarray(likelihood_scales_raw, dtype=np.float64)
    if not np.isfinite(likelihood_scales).all() or bool((likelihood_scales <= 0.0).any()):
        raise ValueError("likelihood_scales must contain positive finite values")
    scale_weights_raw = config.get("likelihood_scale_weights", np.ones(len(likelihood_scales)).tolist())
    scale_weights = np.asarray(scale_weights_raw, dtype=np.float64)
    if len(scale_weights) != len(likelihood_scales) or not np.isfinite(scale_weights).all() or float(scale_weights.sum()) <= 0.0:
        raise ValueError("likelihood_scale_weights must match likelihood_scales and have positive total weight")
    scale_weights /= scale_weights.sum()
    seed_batches = config.get("seed_batches", [int(config.get("seed", 42))])
    if not isinstance(seed_batches, list) or not seed_batches:
        raise ValueError("seed_batches must be a non-empty list")
    batch_predictions: list[np.ndarray] = []
    batch_variances: list[np.ndarray] = []
    batch_log_likelihoods: list[np.ndarray] = []
    mode_levels: list[np.ndarray] = []
    mode_weights: list[np.ndarray] = []
    selector_config = config.get("geometry_selector", {})
    selected_scale: float | None = None
    selected_hold_weight: float | None = None
    selector_code = -1
    if bool(selector_config.get("enabled", False)):
        n_eval = len(target)
        z_values = target["Z"].to_numpy(dtype=float)
        z_span = float(np.nanmax(z_values) - np.nanmin(z_values)) if len(z_values) else 0.0
        n_bin = int(n_eval > int(selector_config.get("n_eval_threshold", 4840)))
        z_thresholds = np.asarray(selector_config.get("z_span_thresholds", [136.73, 185.5133333333342]))
        z_bin = int(np.searchsorted(z_thresholds, z_span, side="right"))
        selector_code = n_bin + 2 * z_bin
        variants = selector_config.get("variants", {})
        variant = variants.get(str(selector_code), selector_config.get("default", {}))
        selected_scale = float(variant.get("likelihood_scale", 8.0))
        selected_hold_weight = float(variant.get("hold_weight", 0.2))
        if not bool(np.isclose(likelihood_scales, selected_scale).any()):
            raise ValueError("geometry selector scale must be included in likelihood_scales")
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
        temperature_predictions: list[np.ndarray] = []
        temperature_variances: list[np.ndarray] = []
        for likelihood_scale in likelihood_scales:
            seed_weights = np.exp(centered_likelihoods / likelihood_scale)
            seed_weights /= seed_weights.sum()
            temperature_prediction = seed_weights @ predictions
            temperature_predictions.append(temperature_prediction)
            temperature_variances.append(
                seed_weights @ np.square(predictions - temperature_prediction[None, :])
            )
        temperature_matrix = np.vstack(temperature_predictions)
        active_scale_weights = scale_weights
        if selected_scale is not None:
            active_scale_weights = np.isclose(likelihood_scales, selected_scale).astype(np.float64)
            active_scale_weights /= active_scale_weights.sum()
        batch_prediction = active_scale_weights @ temperature_matrix
        batch_predictions.append(batch_prediction)
        batch_variances.append(
            active_scale_weights
            @ (
                np.vstack(temperature_variances)
                + np.square(temperature_matrix - batch_prediction[None, :])
            )
        )
        batch_log_likelihoods.append(log_likelihoods)
        mha_config = config.get("mha", {})
        if bool(mha_config.get("enabled", False)):
            mha_scale = float(mha_config.get("likelihood_scale", 5.0))
            mha_weights = np.exp(centered_likelihoods / mha_scale)
            mha_weights /= mha_weights.sum()
            mode_levels.append(predictions.mean(axis=1))
            mode_weights.append(mha_weights / len(seed_batches))

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
    hold_weight = (
        selected_hold_weight
        if selected_hold_weight is not None
        else float(config.get("hold_weight", 0.0))
    )
    prediction = (1.0 - hold_weight) * prediction + hold_weight * anchor_tvt
    mha_shift = 0.0
    mode_separation = 0.0
    minor_mode_mass = 0.0
    if mode_levels:
        mha_shift, mode_separation, minor_mode_mass = _weighted_two_mode_hedge(
            np.concatenate(mode_levels),
            np.concatenate(mode_weights),
            config.get("mha", {}),
        )
        prediction = prediction + mha_shift

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
            "pf_gr_gain": gr_gain,
            "pf_gr_offset": gr_offset,
            "pf_gr_calibrated": gr_calibrated,
            "pf_log_likelihood_spread": float(np.std(all_log_likelihoods)),
            "pf_selected_likelihood_scale": selected_scale if selected_scale is not None else float(np.sum(scale_weights * likelihood_scales)),
            "pf_selector_code": selector_code,
            "pf_hold_weight": hold_weight,
            "pf_mha_shift": mha_shift,
            "pf_mode_separation": mode_separation,
            "pf_minor_mode_mass": minor_mode_mass,
            "y_pred": prediction,
        }
    )
    if "TVT" in target:
        result["y_true"] = target["TVT"].to_numpy(dtype=float)
    return result
