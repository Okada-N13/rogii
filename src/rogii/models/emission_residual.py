from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


GENERIC_FEATURES = (
    "base_prediction",
    "surface_prediction",
    "correction",
    "correction_abs",
    "correction_roughness",
    "correction_cut_mean",
    "correction_cut_std",
    "correction_well_mean_abs",
    "surface_slope",
    "md_fraction",
    "cut_fraction",
)

STACKED_FEATURES = (
    *GENERIC_FEATURES,
    "spatial_minus_standard",
    "typewell_minus_standard",
    "branch_mean_minus_standard",
    "branch_std",
    "learned_entropy",
    "entropy_percentile",
)


def _safe_slope(values: pd.Series, md: pd.Series, groups: pd.Series) -> pd.Series:
    value_diff = values.groupby(groups, sort=False).diff()
    md_diff = md.groupby(groups, sort=False).diff()
    result = value_diff / md_diff.replace(0.0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2.0, 2.0)


def generic_residual_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"cut_id", "well_id", "MD", "cut_fraction", "surface_y_pred", "y_pred"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Residual feature frame is missing columns: {missing}")
    ordered = frame.sort_values(["cut_id", "MD"]).copy()
    correction = ordered["y_pred"].to_numpy(float) - ordered["surface_y_pred"].to_numpy(float)
    ordered["base_prediction"] = ordered["y_pred"].to_numpy(float)
    ordered["surface_prediction"] = ordered["surface_y_pred"].to_numpy(float)
    ordered["correction"] = correction
    ordered["correction_abs"] = np.abs(correction)
    ordered["correction_roughness"] = ordered.groupby("cut_id", sort=False)["correction"].diff().abs().fillna(0.0)
    ordered["correction_cut_mean"] = ordered.groupby("cut_id", sort=False)["correction"].transform("mean")
    ordered["correction_cut_std"] = ordered.groupby("cut_id", sort=False)["correction"].transform("std").fillna(0.0)
    well_mean = ordered.groupby("well_id", sort=False)["correction_abs"].mean()
    ordered["correction_well_mean_abs"] = ordered["well_id"].map(well_mean).to_numpy(float)
    ordered["surface_slope"] = _safe_slope(ordered["surface_y_pred"], ordered["MD"], ordered["cut_id"])
    minimum = ordered.groupby("cut_id", sort=False)["MD"].transform("min")
    span = ordered.groupby("cut_id", sort=False)["MD"].transform("max") - minimum
    ordered["md_fraction"] = ((ordered["MD"] - minimum) / span.replace(0.0, 1.0)).clip(0.0, 1.0)
    output = ordered[list(GENERIC_FEATURES)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return output.sort_index().astype(np.float32)


def stacked_residual_features(
    standard: pd.DataFrame,
    spatial_prediction: np.ndarray,
    typewell_prediction: np.ndarray,
    entropy: np.ndarray,
) -> pd.DataFrame:
    output = generic_residual_features(standard)
    base = standard["y_pred"].to_numpy(float)
    spatial = np.asarray(spatial_prediction, dtype=float)
    typewell = np.asarray(typewell_prediction, dtype=float)
    entropy_values = np.asarray(entropy, dtype=float)
    if not (len(base) == len(spatial) == len(typewell) == len(entropy_values)):
        raise ValueError("Stacked residual features are not row aligned")
    branches = np.column_stack([base, spatial, typewell])
    output["spatial_minus_standard"] = spatial - base
    output["typewell_minus_standard"] = typewell - base
    output["branch_mean_minus_standard"] = branches.mean(axis=1) - base
    output["branch_std"] = branches.std(axis=1)
    output["learned_entropy"] = entropy_values
    output["entropy_percentile"] = pd.Series(entropy_values).rank(method="average", pct=True).to_numpy(float)
    return output[list(STACKED_FEATURES)].astype(np.float32)


def make_residual_model(config: dict[str, Any], seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=float(config.get("learning_rate", 0.04)),
        max_iter=int(config.get("max_iter", 180)),
        max_leaf_nodes=int(config.get("max_leaf_nodes", 15)),
        min_samples_leaf=int(config.get("min_samples_leaf", 500)),
        l2_regularization=float(config.get("l2_regularization", 5.0)),
        max_bins=int(config.get("max_bins", 127)),
        random_state=int(seed),
        early_stopping=False,
    )


def target_invariance_check(frame: pd.DataFrame) -> bool:
    original = generic_residual_features(frame)
    changed = frame.copy()
    changed["y_true"] = changed["y_true"].to_numpy(float) + 991.0
    perturbed = generic_residual_features(changed)
    return bool(np.array_equal(original.to_numpy(), perturbed.to_numpy()))
