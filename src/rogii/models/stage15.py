from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from rogii.data.multicut import TARGET_COLUMNS, feature_columns
from rogii.models.delta_u_surface import REGIONAL_COLUMNS, _make_model, _regional_features


FAMILIES = ("fold", "spatial_fold", "typewell_fold")


def fit_surface_bundle(
    records: pd.DataFrame,
    family: str,
    heldout_fold: int,
    model_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Fit the exact fold-safe Stage 11 surface used for an overlapping test well."""
    train = records[records[family] != int(heldout_fold)].copy()
    if train.empty:
        raise ValueError(f"No surface rows remain for {family}={heldout_fold}")
    regional = dict(model_config.get("regional", {}))
    options = {
        "k_neighbors": int(regional.get("k_neighbors", 16)),
        "distance_floor_ft": float(regional.get("distance_floor_ft", 100.0)),
        "distance_shrink_ft": float(regional.get("distance_shrink_ft", 3000.0)),
    }
    values = _regional_features(train, train, **options)
    for column in REGIONAL_COLUMNS:
        train[column] = values[column]
    columns = feature_columns(train)
    x = train[columns].to_numpy(np.float32)
    models = {}
    # Stage 11 used the completed-fold count in its seed; folds are contiguous 0..N-1.
    model_seed = seed + int(heldout_fold) * 17 + int(heldout_fold)
    for target in TARGET_COLUMNS:
        model = _make_model(model_config, model_seed)
        model.fit(x, train[target].to_numpy(float))
        models[target] = model
    return {
        "family": family,
        "heldout_fold": int(heldout_fold),
        "feature_columns": columns,
        "regional_options": options,
        "donors": train[["well_id", "anchor_x", "anchor_y", *TARGET_COLUMNS]].copy(),
        "models": models,
    }


def predict_surface_coefficients(bundle: dict[str, Any], record: dict[str, Any]) -> pd.Series:
    query = pd.DataFrame([record])
    regional = _regional_features(query, bundle["donors"], **bundle["regional_options"])
    for column in REGIONAL_COLUMNS:
        query[column] = regional[column]
    columns = list(bundle["feature_columns"])
    missing = sorted(set(columns) - set(query.columns))
    if missing:
        raise ValueError(f"Stage 15 surface feature mismatch: {missing}")
    x = query[columns].to_numpy(np.float32)
    for target in TARGET_COLUMNS:
        query[f"pred_{target}"] = bundle["models"][target].predict(x)
    return query.iloc[0]


def save_surface_bundle(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_surface_bundle(path: Path) -> dict[str, Any]:
    return joblib.load(path)
