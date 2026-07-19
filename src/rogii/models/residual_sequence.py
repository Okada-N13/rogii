from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rogii.data.loading import load_horizontal_well, load_typewell
from rogii.models.trend import predict_guarded_trend


@dataclass(frozen=True)
class ResidualFeatures:
    frame: pd.DataFrame
    columns: list[str]
    sampled: np.ndarray


def _gradient(values: np.ndarray, md: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(len(values), dtype=np.float64)
    safe_md = md.copy()
    for index in range(1, len(safe_md)):
        if safe_md[index] <= safe_md[index - 1]:
            safe_md[index] = safe_md[index - 1] + 1.0
    return np.gradient(values, safe_md)


def _finite(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)


def _sample_positions(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, maximum, dtype=np.int64))


def build_residual_features(
    base_oof: pd.DataFrame,
    horizontal_paths: dict[str, Path],
    windows: list[int],
    max_rows_per_well: int,
) -> ResidualFeatures:
    required = {
        "id",
        "well_id",
        "row_index",
        "MD",
        "Z",
        "anchor_value",
        "surface_anchor",
        "pf_seed_std",
        "pf_gr_sigma",
        "pf_log_likelihood_spread",
        "y_pred",
    }
    missing = sorted(required - set(base_oof.columns))
    if missing:
        raise ValueError(f"Base OOF is missing Stage 2 columns: {missing}")
    if max_rows_per_well < 2:
        raise ValueError("max_rows_per_well must be at least two")

    feature_frames: list[pd.DataFrame] = []
    sampled_parts: list[np.ndarray] = []
    offset = 0
    feature_columns: list[str] | None = None
    for well_id, prediction in base_oof.groupby("well_id", sort=False):
        prediction = prediction.sort_values("row_index").reset_index(drop=True)
        path = horizontal_paths.get(str(well_id))
        if path is None:
            raise FileNotFoundError(f"No horizontal CSV for Stage 2 well {well_id}")
        horizontal = load_horizontal_well(path)
        rows = prediction["row_index"].to_numpy(dtype=np.int64)
        if rows.min() < 0 or rows.max() >= len(horizontal):
            raise ValueError(f"{well_id}: Stage 2 row_index is outside the horizontal CSV")
        target = horizontal.iloc[rows]
        if not np.allclose(target["MD"].to_numpy(dtype=float), prediction["MD"].to_numpy(dtype=float)):
            raise ValueError(f"{well_id}: Stage 2 rows do not align with the horizontal CSV")

        full_md = horizontal["MD"].to_numpy(dtype=np.float64)
        full_gr_series = pd.to_numeric(horizontal["GR"], errors="coerce").interpolate(limit_direction="both")
        full_gr = full_gr_series.fillna(float(full_gr_series.mean())).to_numpy(dtype=np.float64)
        gr_median = float(np.median(full_gr))
        gr_scale = float(np.clip(np.median(np.abs(full_gr - gr_median)) * 1.4826, 5.0, 60.0))
        target_md = prediction["MD"].to_numpy(dtype=np.float64)
        target_z = prediction["Z"].to_numpy(dtype=np.float64)
        target_gr = full_gr[rows]
        base_prediction = prediction["y_pred"].to_numpy(dtype=np.float64)
        predicted_surface = base_prediction + target_z

        typewell = load_typewell(path)
        typewell_gr = typewell["GR"].interpolate(limit_direction="both").fillna(float(typewell["GR"].mean()))
        expected_gr = np.interp(
            base_prediction,
            typewell["TVT"].to_numpy(dtype=float),
            typewell_gr.to_numpy(dtype=float),
        )
        mismatch = (target_gr - expected_gr) / float(prediction["pf_gr_sigma"].iloc[0])
        horizon = target_md - target_md[0]
        horizon_scale = max(float(horizon[-1]), 1.0)

        trend_predictions: dict[str, np.ndarray] = {}
        trend_diagnostics: dict[str, float] = {}
        for window_ft in (100, 500, 1000):
            trend = predict_guarded_trend(
                horizontal,
                {
                    "target": "surface",
                    "degree": 1,
                    "window_ft": window_ft,
                    "minimum_points": 20,
                    "ridge": 0.001,
                    "huber_delta": 1.5,
                    "robust_iterations": 5,
                    "slope_clip": 1.2,
                    "max_delta_ft": 250,
                },
            )
            trend_predictions[f"linear_{window_ft}"] = trend["y_pred"].to_numpy(dtype=float)
            trend_diagnostics[f"linear_slope_{window_ft}"] = float(trend["trend_slope"].iloc[0])
        quadratic = predict_guarded_trend(
            horizontal,
            {
                "target": "surface",
                "degree": 2,
                "window_ft": 1000,
                "minimum_points": 30,
                "ridge": 0.01,
                "huber_delta": 1.5,
                "robust_iterations": 5,
                "slope_clip": 1.2,
                "curvature_clip": 0.0001,
                "max_delta_ft": 250,
                "strength": 0.5,
            },
        )
        trend_predictions["quadratic_1000"] = quadratic["y_pred"].to_numpy(dtype=float)
        trend_diagnostics["quadratic_curvature_1000"] = float(quadratic["trend_curvature"].iloc[0])

        features: dict[str, np.ndarray] = {
            "horizon_kft": horizon / 1000.0,
            "horizon_fraction": horizon / horizon_scale,
            "well_length_kft": np.full(len(prediction), horizon_scale / 1000.0),
            "z_delta_kft": (target_z - target_z[0]) / 1000.0,
            "z_slope": _gradient(target_z, target_md),
            "gr_robust_z": (target_gr - gr_median) / gr_scale,
            "gr_gradient": _gradient(target_gr, target_md) / gr_scale,
            "pf_surface_delta_100ft": (
                predicted_surface - prediction["surface_anchor"].to_numpy(dtype=float)
            )
            / 100.0,
            "pf_surface_slope": _gradient(predicted_surface, target_md),
            "pf_tvt_delta_100ft": (
                base_prediction - prediction["anchor_value"].to_numpy(dtype=float)
            )
            / 100.0,
            "pf_tvt_slope": _gradient(base_prediction, target_md),
            "pf_seed_std_10ft": prediction["pf_seed_std"].to_numpy(dtype=float) / 10.0,
            "pf_gr_sigma_30": prediction["pf_gr_sigma"].to_numpy(dtype=float) / 30.0,
            "pf_loglik_spread": np.log1p(
                np.maximum(prediction["pf_log_likelihood_spread"].to_numpy(dtype=float), 0.0)
            ),
            "typewell_gr_mismatch": mismatch,
            "typewell_gr_abs_mismatch": np.abs(mismatch),
            "anchor_minus_pf_100ft": (
                prediction["anchor_value"].to_numpy(dtype=float) - base_prediction
            )
            / 100.0,
            "flat_surface_minus_pf_100ft": (
                prediction["surface_anchor"].to_numpy(dtype=float) - target_z - base_prediction
            )
            / 100.0,
        }
        for name, values in trend_predictions.items():
            features[f"{name}_minus_pf_100ft"] = (values - base_prediction) / 100.0
        for name, value in trend_diagnostics.items():
            features[name] = np.full(len(prediction), value)
        gr_series = pd.Series(full_gr)
        for window in windows:
            if window < 3:
                raise ValueError("sequence windows must be at least three")
            rolling = gr_series.rolling(window, center=True, min_periods=1)
            rolling_mean = rolling.mean().to_numpy(dtype=float)[rows]
            rolling_std = rolling.std(ddof=0).fillna(0.0).to_numpy(dtype=float)[rows]
            features[f"gr_mean_delta_w{window}"] = (target_gr - rolling_mean) / gr_scale
            features[f"gr_std_w{window}"] = rolling_std / gr_scale

        if feature_columns is None:
            feature_columns = list(features)
        frame = prediction.copy()
        for name, values in features.items():
            frame[name] = _finite(values)
        frame["base_y_pred"] = frame["y_pred"].to_numpy(dtype=float)
        if "y_true" in frame:
            frame["residual_target"] = frame["y_true"] - frame["base_y_pred"]
        feature_frames.append(frame)
        local_sample = _sample_positions(len(frame), max_rows_per_well)
        sampled_parts.append(offset + local_sample)
        offset += len(frame)

    assert feature_columns is not None
    combined = pd.concat(feature_frames, ignore_index=True)
    sampled = np.concatenate(sampled_parts)
    return ResidualFeatures(frame=combined, columns=feature_columns, sampled=sampled)


def make_residual_model(config: dict[str, Any], seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=str(config.get("loss", "squared_error")),
        learning_rate=float(config.get("learning_rate", 0.04)),
        max_iter=int(config.get("max_iter", 160)),
        max_leaf_nodes=int(config.get("max_leaf_nodes", 15)),
        min_samples_leaf=int(config.get("min_samples_leaf", 40)),
        l2_regularization=float(config.get("l2_regularization", 10.0)),
        max_bins=int(config.get("max_bins", 127)),
        random_state=seed,
    )


def sampled_training_data(
    features: ResidualFeatures,
    excluded_fold: int | None,
    target_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = features.frame
    if "residual_target" not in frame or "fold" not in frame:
        raise ValueError("Training residual features require y_true and fold")
    indices = features.sampled
    if excluded_fold is not None:
        indices = indices[frame.iloc[indices]["fold"].to_numpy(dtype=int) != excluded_fold]
    sampled = frame.iloc[indices]
    x = sampled[features.columns].to_numpy(dtype=np.float32)
    y = np.clip(sampled["residual_target"].to_numpy(dtype=float), -target_clip, target_clip)
    well_lengths = frame.groupby("well_id", sort=False).size()
    sampled_lengths = sampled.groupby("well_id", sort=False).size()
    weights = sampled["well_id"].map(well_lengths / sampled_lengths).to_numpy(dtype=float)
    return x, y, weights
