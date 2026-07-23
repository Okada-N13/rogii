from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


FIELDS = (
    "value", "feature_idx", "num_threshold", "missing_go_to_left",
    "left", "right", "is_leaf", "is_categorical",
)


def export_hist_gradient_boosting(model: Any, path: Path) -> None:
    """Export a fitted sklearn HGB regressor to a version-independent NPZ."""
    arrays: dict[str, np.ndarray] = {
        "baseline": np.asarray(model._baseline_prediction, dtype=np.float64).reshape(1),
        "n_features": np.asarray([model.n_features_in_], dtype=np.int32),
        "n_trees": np.asarray([len(model._predictors)], dtype=np.int32),
    }
    for tree_index, stage in enumerate(model._predictors):
        if len(stage) != 1:
            raise ValueError("Only scalar HistGradientBoostingRegressor models are supported")
        nodes = stage[0].nodes
        if np.any(nodes["is_categorical"]):
            raise ValueError("Categorical HGB splits are not supported")
        for field in FIELDS:
            arrays[f"tree_{tree_index:03d}__{field}"] = np.asarray(nodes[field])
    np.savez_compressed(path, **arrays)


def load_portable_hgb(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        n_trees = int(archive["n_trees"][0])
        trees = [
            {field: archive[f"tree_{index:03d}__{field}"].copy() for field in FIELDS}
            for index in range(n_trees)
        ]
        return {
            "baseline": float(archive["baseline"][0]),
            "n_features": int(archive["n_features"][0]),
            "trees": trees,
        }


def predict_portable_hgb(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != int(model["n_features"]):
        raise ValueError("Portable HGB feature shape mismatch")
    output = np.full(len(values), float(model["baseline"]), dtype=np.float64)
    for tree in model["trees"]:
        for row_index, row in enumerate(values):
            node = 0
            while not bool(tree["is_leaf"][node]):
                feature = int(tree["feature_idx"][node])
                value = row[feature]
                go_left = (
                    bool(tree["missing_go_to_left"][node])
                    if np.isnan(value)
                    else value <= float(tree["num_threshold"][node])
                )
                node = int(tree["left"][node] if go_left else tree["right"][node])
            output[row_index] += float(tree["value"][node])
    return output
