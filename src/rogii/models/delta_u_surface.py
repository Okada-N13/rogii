from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rogii.data.multicut import TARGET_COLUMNS, feature_columns


REGIONAL_COLUMNS = [
    "regional_slope_correction",
    "regional_curvature",
    "regional_nearest_distance_ft",
    "regional_reliability",
]


def _well_targets(records: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "anchor_x": "median",
        "anchor_y": "median",
        "target_slope_correction": "median",
        "target_curvature": "median",
    }
    return records.groupby("well_id", as_index=False).agg(aggregations)


def _regional_features(
    query: pd.DataFrame,
    donors: pd.DataFrame,
    *,
    k_neighbors: int,
    distance_floor_ft: float,
    distance_shrink_ft: float,
) -> pd.DataFrame:
    donor_wells = _well_targets(donors)
    donor_xy = donor_wells[["anchor_x", "anchor_y"]].to_numpy(dtype=float)
    donor_targets = donor_wells[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    outputs: list[dict[str, float]] = []
    for row in query.itertuples(index=False):
        mask = donor_wells["well_id"].astype(str).to_numpy() != str(row.well_id)
        xy = donor_xy[mask]
        targets = donor_targets[mask]
        if len(xy) == 0:
            raise ValueError("No fold-local regional donors remain")
        distances = np.sqrt(
            np.square(xy[:, 0] - float(row.anchor_x))
            + np.square(xy[:, 1] - float(row.anchor_y))
        )
        take = min(int(k_neighbors), len(distances))
        nearest_indices = np.argpartition(distances, take - 1)[:take]
        nearest_distances = distances[nearest_indices]
        weights = 1.0 / np.maximum(nearest_distances, float(distance_floor_ft))
        weights /= weights.sum()
        local = np.sum(weights[:, None] * targets[nearest_indices], axis=0)
        global_target = np.mean(targets, axis=0)
        nearest = float(np.min(nearest_distances))
        reliability = float(distance_shrink_ft / (distance_shrink_ft + nearest))
        prediction = global_target + reliability * (local - global_target)
        outputs.append(
            {
                "regional_slope_correction": float(prediction[0]),
                "regional_curvature": float(prediction[1]),
                "regional_nearest_distance_ft": nearest,
                "regional_reliability": reliability,
            }
        )
    return pd.DataFrame.from_records(outputs, index=query.index)


def _make_model(config: dict[str, Any], seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=float(config.get("learning_rate", 0.05)),
        max_iter=int(config.get("max_iter", 300)),
        max_leaf_nodes=int(config.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(config.get("min_samples_leaf", 20)),
        l2_regularization=float(config.get("l2_regularization", 1.0)),
        max_bins=int(config.get("max_bins", 127)),
        random_state=int(seed),
        early_stopping=False,
    )


def crossfit_delta_u_coefficients(
    records: pd.DataFrame,
    fold_column: str,
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    if fold_column not in records:
        raise ValueError(f"records are missing {fold_column}")
    regional = dict(config.get("regional", {}))
    outputs: list[pd.DataFrame] = []
    resolved_features: list[str] | None = None
    for fold in sorted(int(value) for value in records[fold_column].unique()):
        train = records[records[fold_column] != fold].copy()
        valid = records[records[fold_column] == fold].copy()
        train_region = _regional_features(
            train,
            train,
            k_neighbors=int(regional.get("k_neighbors", 16)),
            distance_floor_ft=float(regional.get("distance_floor_ft", 100.0)),
            distance_shrink_ft=float(regional.get("distance_shrink_ft", 3000.0)),
        )
        valid_region = _regional_features(
            valid,
            train,
            k_neighbors=int(regional.get("k_neighbors", 16)),
            distance_floor_ft=float(regional.get("distance_floor_ft", 100.0)),
            distance_shrink_ft=float(regional.get("distance_shrink_ft", 3000.0)),
        )
        for column in REGIONAL_COLUMNS:
            train[column] = train_region[column]
            valid[column] = valid_region[column]
        columns = feature_columns(train)
        if resolved_features is None:
            resolved_features = columns
        elif columns != resolved_features:
            raise RuntimeError("Feature schema changed across folds")
        x_train = train[columns].to_numpy(dtype=np.float32)
        x_valid = valid[columns].to_numpy(dtype=np.float32)
        for target in TARGET_COLUMNS:
            model = _make_model(config, seed + fold * 17 + len(outputs))
            model.fit(x_train, train[target].to_numpy(dtype=float))
            valid[f"pred_{target}"] = model.predict(x_valid)
        outputs.append(valid)
    result = pd.concat(outputs, ignore_index=True)
    if result["cut_id"].duplicated().any() or len(result) != len(records):
        raise RuntimeError("Cross-fit coefficient output is not one-to-one with cuts")
    return result, list(resolved_features or [])


def build_row_predictions(
    coefficient_oof: pd.DataFrame,
    horizontal_by_well: dict[str, pd.DataFrame],
    fold_column: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    max_rows = int(config.get("max_eval_rows_per_cut", 1024))
    correction_cap = float(config.get("correction_cap_ft", 50.0))
    correction_weight = float(config.get("correction_weight", 0.35))
    slope_cap = float(config.get("slope_correction_cap", 80.0))
    curvature_cap = float(config.get("curvature_cap", 30.0))
    outputs: list[pd.DataFrame] = []
    for row in coefficient_oof.itertuples(index=False):
        horizontal = horizontal_by_well[str(row.well_id)]
        positions = np.arange(int(row.cut_index), len(horizontal), dtype=np.int64)
        if max_rows > 0 and len(positions) > max_rows:
            positions = np.unique(np.linspace(positions[0], positions[-1], max_rows).round().astype(int))
        suffix = horizontal.iloc[positions]
        horizon_kft = (suffix["MD"].to_numpy(dtype=float) - float(row.anchor_md)) / 1000.0
        baseline_u = float(row.anchor_u) + float(row.prefix_u_slope_per_kft) * horizon_kft
        slope = float(np.clip(row.pred_target_slope_correction, -slope_cap, slope_cap))
        curvature = float(np.clip(row.pred_target_curvature, -curvature_cap, curvature_cap))
        correction = correction_weight * np.clip(
            slope * horizon_kft + curvature * np.square(horizon_kft),
            -correction_cap,
            correction_cap,
        )
        z = suffix["Z"].to_numpy(dtype=float)
        outputs.append(
            pd.DataFrame(
                {
                    "id": str(row.cut_id) + "_" + suffix["row_index"].astype(str),
                    "well_id": str(row.well_id),
                    "cut_id": str(row.cut_id),
                    "row_index": suffix["row_index"].to_numpy(dtype=np.int64),
                    "MD": suffix["MD"].to_numpy(dtype=float),
                    "Z": z,
                    "y_true": suffix["TVT"].to_numpy(dtype=float),
                    "base_y_pred": baseline_u - z,
                    "surface_correction": correction,
                    "y_pred": baseline_u + correction - z,
                    "fold": int(getattr(row, fold_column)),
                }
            )
        )
    return pd.concat(outputs, ignore_index=True)
