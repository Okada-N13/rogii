from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def make_spatial_blocks(wells: pd.DataFrame, n_blocks: int, seed: int) -> pd.Series:
    if n_blocks < 2 or n_blocks > len(wells):
        raise ValueError("n_blocks must be between two and the number of wells")
    coordinates = wells[["x", "y"]].to_numpy(dtype=float)
    labels = KMeans(n_clusters=n_blocks, random_state=seed, n_init=20).fit_predict(coordinates)
    return pd.Series(labels.astype(np.int16), index=wells.index, name="spatial_fold")


def fit_well_residual_targets(base_oof: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for well_id, frame in base_oof.groupby("well_id", sort=False):
        ordered = frame.sort_values("MD")
        horizon_kft = (
            ordered["MD"].to_numpy(dtype=float) - float(ordered["MD"].iloc[0])
        ) / 1000.0
        residual = ordered["y_true"].to_numpy(dtype=float) - ordered["y_pred"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(ordered)), horizon_kft])
        coefficients = np.linalg.lstsq(design, residual, rcond=None)[0]
        records.append(
            {
                "well_id": str(well_id),
                "residual_bias": float(coefficients[0]),
                "residual_slope_per_kft": float(coefficients[1]),
                "residual_mean": float(residual.mean()),
                "n_rows": len(ordered),
                "fold": int(ordered["fold"].iloc[0]),
            }
        )
    return pd.DataFrame.from_records(records)


def spatial_knn_predictions(
    wells: pd.DataFrame,
    fold_column: str,
    config: dict[str, Any],
    shuffle_targets: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    required = {"well_id", "x", "y", "residual_bias", "residual_slope_per_kft", fold_column}
    missing = sorted(required - set(wells.columns))
    if missing:
        raise ValueError(f"Spatial wells are missing columns: {missing}")
    k_neighbors = int(config.get("k_neighbors", 16))
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")
    distance_floor = float(config.get("distance_floor_ft", 100.0))
    distance_scale = float(config.get("distance_shrink_ft", 3000.0))
    distance_power = float(config.get("distance_power", 1.0))
    bias_clip = float(config.get("bias_clip_ft", 20.0))
    slope_clip = float(config.get("slope_clip_ft_per_kft", 15.0))
    rng = np.random.default_rng(seed)
    outputs: list[pd.DataFrame] = []
    for fold in sorted(int(value) for value in wells[fold_column].unique()):
        train = wells[wells[fold_column] != fold].copy()
        valid = wells[wells[fold_column] == fold].copy()
        targets = train[["residual_bias", "residual_slope_per_kft"]].to_numpy(dtype=float)
        if shuffle_targets:
            targets = targets[rng.permutation(len(targets))]
        train_xy = train[["x", "y"]].to_numpy(dtype=float)
        valid_xy = valid[["x", "y"]].to_numpy(dtype=float)
        distances = np.sqrt(np.square(valid_xy[:, None, :] - train_xy[None, :, :]).sum(axis=2))
        take = min(k_neighbors, len(train))
        neighbors = np.argpartition(distances, take - 1, axis=1)[:, :take]
        neighbor_distances = np.take_along_axis(distances, neighbors, axis=1)
        order = np.argsort(neighbor_distances, axis=1)
        neighbors = np.take_along_axis(neighbors, order, axis=1)
        neighbor_distances = np.take_along_axis(neighbor_distances, order, axis=1)
        weights = 1.0 / np.power(np.maximum(neighbor_distances, distance_floor), distance_power)
        weights /= weights.sum(axis=1, keepdims=True)
        neighbor_targets = targets[neighbors]
        local = np.sum(weights[:, :, None] * neighbor_targets, axis=1)
        global_target = np.average(
            targets,
            axis=0,
            weights=train["n_rows"].to_numpy(dtype=float),
        )
        nearest = neighbor_distances[:, 0]
        reliability = distance_scale / (distance_scale + nearest)
        prediction = global_target[None, :] + reliability[:, None] * (
            local - global_target[None, :]
        )
        prediction[:, 0] = np.clip(prediction[:, 0], -bias_clip, bias_clip)
        prediction[:, 1] = np.clip(prediction[:, 1], -slope_clip, slope_clip)
        valid["spatial_bias_pred"] = prediction[:, 0]
        valid["spatial_slope_pred_per_kft"] = prediction[:, 1]
        valid["spatial_nearest_distance_ft"] = nearest
        valid["spatial_reliability"] = reliability
        outputs.append(valid)
    return pd.concat(outputs, ignore_index=True)


def apply_spatial_correction(
    base_oof: pd.DataFrame,
    well_predictions: pd.DataFrame,
    blend_weight: float,
    correction_cap: float,
    output_fold: str,
) -> pd.DataFrame:
    columns = [
        "well_id",
        "spatial_bias_pred",
        "spatial_slope_pred_per_kft",
        "spatial_nearest_distance_ft",
        "spatial_reliability",
        output_fold,
    ]
    result = base_oof.drop(columns=["fold"], errors="ignore").merge(
        well_predictions[columns], on="well_id", how="left", validate="many_to_one"
    )
    first_md = result.groupby("well_id")["MD"].transform("min")
    horizon_kft = (result["MD"] - first_md) / 1000.0
    raw_correction = result["spatial_bias_pred"] + result[
        "spatial_slope_pred_per_kft"
    ] * horizon_kft
    result["spatial_correction"] = np.clip(raw_correction, -correction_cap, correction_cap)
    result["base_y_pred"] = result["y_pred"]
    result["y_pred"] = result["base_y_pred"] + blend_weight * result["spatial_correction"]
    result["fold"] = result[output_fold].astype(np.int16)
    return result
