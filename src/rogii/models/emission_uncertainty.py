from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def uncertainty_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"cut_id", "well_id", "row_index", "surface_y_pred", "y_pred"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Uncertainty frame is missing columns: {missing}")
    ordered = frame.sort_values(["cut_id", "row_index"]).copy()
    correction = ordered["y_pred"].to_numpy(float) - ordered["surface_y_pred"].to_numpy(float)
    ordered["correction"] = correction
    ordered["correction_abs"] = np.abs(correction)
    ordered["correction_roughness"] = (
        ordered.groupby("cut_id", sort=False)["correction"].diff().abs().fillna(0.0)
    )
    well_risk = ordered.groupby("well_id")["correction_abs"].mean()
    ordered["well_correction_abs"] = ordered["well_id"].map(well_risk).to_numpy(float)
    for column in ("correction_abs", "correction_roughness", "well_correction_abs"):
        ordered[f"{column}_pct"] = ordered[column].rank(method="average", pct=True).to_numpy(float)
    ordered["local_risk"] = ordered[
        ["correction_abs_pct", "correction_roughness_pct", "well_correction_abs_pct"]
    ].mean(axis=1)
    return ordered.sort_index()


def apply_uncertainty_profile(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    profile: dict[str, Any],
) -> np.ndarray:
    surface = frame["surface_y_pred"].to_numpy(float)
    correction = frame["y_pred"].to_numpy(float) - surface
    cap = float(profile.get("correction_cap", 1e9))
    correction = np.clip(correction, -cap, cap)
    correction *= float(profile.get("global_scale", 1.0))
    threshold = float(profile.get("risk_threshold", 1.1))
    scale = float(profile.get("high_risk_scale", 1.0))
    high_risk = features["local_risk"].to_numpy(float) >= threshold
    correction[high_risk] *= scale
    return surface + correction


def prediction_error_correlation(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    truth = frame["y_true"].to_numpy(float)
    output: dict[str, float] = {}
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            left_error = frame[left].to_numpy(float) - truth
            right_error = frame[right].to_numpy(float) - truth
            output[f"{left}__{right}"] = float(np.corrcoef(left_error, right_error)[0, 1])
    return output
