from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rogii.models.anchor import predict_anchor
from rogii.models.particle_filter import predict_particle_filter
from rogii.models.trend import predict_guarded_trend


def predict_model(
    frame: pd.DataFrame,
    config: dict[str, Any],
    typewell: pd.DataFrame | None = None,
) -> pd.DataFrame:
    name = str(config.get("name"))
    if name in {"last_tvt_anchor", "flat_surface_anchor"}:
        return predict_anchor(frame, mode=name)
    if name == "guarded_trend":
        return predict_guarded_trend(frame, config)
    if name == "particle_filter":
        if typewell is None:
            raise ValueError("particle_filter requires a companion typewell")
        return predict_particle_filter(frame, typewell, config)
    if name == "fixed_blend":
        components = config.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError("fixed_blend requires a non-empty components list")
        weights = np.asarray([float(component["weight"]) for component in components], dtype=float)
        if (weights < 0).any() or not np.isclose(weights.sum(), 1.0, atol=1e-8):
            raise ValueError("fixed_blend weights must be non-negative and sum to one")
        predictions = [predict_model(frame, dict(component["model"]), typewell) for component in components]
        reference_ids = predictions[0]["id"].to_numpy()
        for prediction in predictions[1:]:
            if not np.array_equal(reference_ids, prediction["id"].to_numpy()):
                raise RuntimeError("Blend components produced different target rows")
        result = predictions[0].copy()
        matrix = np.vstack([prediction["y_pred"].to_numpy(dtype=float) for prediction in predictions])
        result["model"] = "fixed_blend"
        result["y_pred"] = weights @ matrix
        for column in ("trend_slope", "trend_curvature"):
            result.drop(columns=column, errors="ignore", inplace=True)
        return result
    raise ValueError(f"Unsupported model: {name!r}")
