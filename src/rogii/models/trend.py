from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rogii.data.schema import target_mask


def _anchored_robust_fit(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    ridge: float,
    huber_delta: float,
    iterations: int,
) -> tuple[np.ndarray, float]:
    scale = max(float(np.max(np.abs(x))), 1.0)
    normalized = x / scale
    design = np.column_stack([normalized ** power for power in range(1, degree + 1)])
    weights = np.ones(len(x), dtype=float)
    penalty = np.eye(degree, dtype=float) * ridge
    coefficients = np.zeros(degree, dtype=float)
    for _ in range(max(1, iterations)):
        weighted_design = design * weights[:, None]
        lhs = design.T @ weighted_design + penalty
        rhs = design.T @ (weights * y)
        coefficients = np.linalg.solve(lhs, rhs)
        residual = y - design @ coefficients
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        robust_scale = max(1.4826 * mad, 1e-6)
        standardized = np.abs(residual - center) / robust_scale
        weights = np.minimum(1.0, huber_delta / np.maximum(standardized, 1e-12))
    return coefficients, scale


def predict_guarded_trend(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    hidden = target_mask(frame).to_numpy()
    if not hidden.any() or hidden.all():
        raise ValueError("TVT_input must contain a known prefix and hidden suffix")
    first_hidden = int(np.flatnonzero(hidden)[0])
    known = frame.iloc[:first_hidden]
    target = frame.iloc[first_hidden:]
    anchor = known.iloc[-1]
    anchor_md = float(anchor["MD"])
    anchor_tvt = float(anchor["TVT_input"])
    anchor_surface = float(anchor_tvt + anchor["Z"])

    target_space = str(config.get("target", "tvt"))
    if target_space == "tvt":
        known_values = known["TVT_input"].to_numpy(dtype=float)
        anchor_value = anchor_tvt
    elif target_space == "surface":
        known_values = known["TVT_input"].to_numpy(dtype=float) + known["Z"].to_numpy(dtype=float)
        anchor_value = anchor_surface
    else:
        raise ValueError(f"trend target must be tvt or surface, got {target_space!r}")

    window_ft = float(config.get("window_ft", 500.0))
    window_mask = known["MD"].to_numpy(dtype=float) >= anchor_md - window_ft
    minimum_points = int(config.get("minimum_points", 20))
    if int(window_mask.sum()) < minimum_points:
        window_mask = np.ones(len(known), dtype=bool)
    x = known.loc[window_mask, "MD"].to_numpy(dtype=float) - anchor_md
    y = known_values[window_mask] - anchor_value
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < minimum_points:
        raise ValueError(f"Not enough finite prefix points for trend fit: {len(x)}")

    degree = int(config.get("degree", 1))
    if degree not in {1, 2}:
        raise ValueError("Only linear and quadratic trends are supported")
    coefficients, scale = _anchored_robust_fit(
        x=x,
        y=y,
        degree=degree,
        ridge=float(config.get("ridge", 1e-3)),
        huber_delta=float(config.get("huber_delta", 1.5)),
        iterations=int(config.get("robust_iterations", 5)),
    )

    physical_slope = float(coefficients[0] / scale)
    slope_clip = config.get("slope_clip")
    if slope_clip is not None:
        physical_slope = float(np.clip(physical_slope, -float(slope_clip), float(slope_clip)))
    physical_curvature = 0.0
    if degree == 2:
        physical_curvature = float(coefficients[1] / (scale * scale))
        curvature_clip = config.get("curvature_clip")
        if curvature_clip is not None:
            physical_curvature = float(
                np.clip(physical_curvature, -float(curvature_clip), float(curvature_clip))
            )

    horizon = target["MD"].to_numpy(dtype=float) - anchor_md
    delta = physical_slope * horizon + physical_curvature * np.square(horizon)
    strength = float(config.get("strength", 1.0))
    decay_ft = config.get("decay_ft")
    if decay_ft is not None:
        decay = float(decay_ft)
        horizon_strength = np.minimum(1.0, decay / np.maximum(horizon, decay))
    else:
        horizon_strength = 1.0
    delta = delta * strength * horizon_strength
    max_delta_ft = config.get("max_delta_ft")
    if max_delta_ft is not None:
        delta = np.clip(delta, -float(max_delta_ft), float(max_delta_ft))

    predicted_target = anchor_value + delta
    if target_space == "surface":
        prediction = predicted_target - target["Z"].to_numpy(dtype=float)
    else:
        prediction = predicted_target

    model_label = f"{target_space}_degree{degree}_trend"
    result = pd.DataFrame(
        {
            "id": target["well_id"].astype(str) + "_" + target["row_index"].astype(str),
            "well_id": target["well_id"].astype(str).to_numpy(),
            "row_index": target["row_index"].to_numpy(dtype=np.int64),
            "MD": target["MD"].to_numpy(dtype=float),
            "Z": target["Z"].to_numpy(dtype=float),
            "model": model_label,
            "anchor_value": anchor_value,
            "surface_anchor": anchor_surface,
            "trend_slope": physical_slope,
            "trend_curvature": physical_curvature,
            "y_pred": prediction,
        }
    )
    if "TVT" in target:
        result["y_true"] = target["TVT"].to_numpy(dtype=float)
    return result

