from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


DEFAULT_PUBLIC_FEATURES = [
    "last_known_tvt",
    "pf_ancc",
    "pf_ancc_std",
    "pf_ancc_delta",
    "pf_z",
    "pf_z_delta",
    "pf_vs_z",
    "beam_cons_d",
    "beam_loose_d",
    "beam_vcons_d",
    "beam_sm5_d",
    "beam_vloose_d",
    "beam_mid_d",
    "beam_stiff_d",
    "beam_mean_d",
    "beam_std_d",
    "beam_med_d",
    "sc8_d",
    "sc8_sc",
    "sc15_d",
    "sc15_sc",
    "sc25_d",
    "sc25_sc",
    "sc_cons_d",
    "sc_ens_d",
    "sc_trust",
    "hyb_d",
    "sig_std",
    "sig_mean_d",
    "form_mean_d",
    "form_std_d",
    "form_rng_d",
    "spatial_knn_dist",
    "dense_std",
    "dense_dist",
    "dense_rmse",
    "dense_bias",
    "dense_nb_std",
    "pf_vs_spatial",
    "pf_vs_dense",
    "spatial_vs_dense",
    "beam_vs_spatial",
    "sc_vs_beam",
    "cal_a",
    "cal_b",
    "pfx_rmse",
    "known_len",
    "eval_len",
    "slp_all",
    "slp_50",
    "slp_z",
    "md_since",
    "frac",
    "frac2",
    "sqrt_frac",
    "z",
    "dx",
    "dy",
    "dz",
    "dxy",
    "dzdmd",
    "dxdmd",
    "dydmd",
    "gr",
    "gr_d1",
    "gr_d2",
    "gr_env",
    "gr_nrg",
    "gr_vs_tw_anc",
    "gr_vs_slp_all",
    "likpf_mean_d",
]


@dataclass(frozen=True)
class PublicResidualFeatures:
    frame: pd.DataFrame
    columns: list[str]
    sampled: np.ndarray


def assign_group_folds(groups: pd.Series, n_splits: int) -> np.ndarray:
    values = groups.astype(str).to_numpy()
    folds = np.empty(len(values), dtype=np.int16)
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(values), dtype=np.float32)
    for fold, (_, valid) in enumerate(splitter.split(dummy, dummy, groups=values)):
        folds[valid] = fold
    return folds


def crossfit_positive_ridge(
    base_oof: np.ndarray,
    target_delta: np.ndarray,
    groups: pd.Series,
    n_splits: int,
    alpha: float,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(base_oof, dtype=np.float64)
    y = np.asarray(target_delta, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("base_oof must be a two-dimensional array aligned with target_delta")
    folds = assign_group_folds(groups, n_splits)
    prediction = np.empty(len(y), dtype=np.float64)
    for fold in range(n_splits):
        valid = folds == fold
        model = Ridge(alpha=alpha, tol=tol, positive=True, fit_intercept=True)
        model.fit(x[~valid], y[~valid])
        prediction[valid] = model.predict(x[valid])
    return prediction, folds


def apply_public_delta_postprocess(
    frame: pd.DataFrame,
    model_delta: np.ndarray,
    alpha: float,
    tau_md: float,
    pf_weight: float,
) -> np.ndarray:
    required = {"last_known_tvt", "pf_ancc", "md_since"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Public artifact frame is missing columns: {missing}")
    last = frame["last_known_tvt"].to_numpy(dtype=np.float64)
    pf_delta = frame["pf_ancc"].to_numpy(dtype=np.float64) - last
    delta = np.asarray(model_delta, dtype=np.float64) * (1.0 - pf_weight) + pf_delta * pf_weight
    if tau_md > 0:
        md_since = np.maximum(frame["md_since"].to_numpy(dtype=np.float64), 0.0)
        delta *= 1.0 - np.exp(-md_since / tau_md)
    return last + alpha * delta


def _sample_by_well(frame: pd.DataFrame, maximum: int) -> np.ndarray:
    if maximum < 2:
        raise ValueError("max_rows_per_well must be at least two")
    indices: list[np.ndarray] = []
    for _, well in frame.groupby("well_id", sort=False):
        positions = well.index.to_numpy(dtype=np.int64)
        if len(positions) > maximum:
            positions = positions[np.unique(np.linspace(0, len(positions) - 1, maximum, dtype=np.int64))]
        indices.append(positions)
    return np.concatenate(indices)


def build_public_residual_features(
    frame: pd.DataFrame,
    base_model_oof: np.ndarray,
    base_prediction: np.ndarray,
    requested_columns: list[str] | None,
    max_rows_per_well: int,
) -> PublicResidualFeatures:
    required = {"well_id", "target", "last_known_tvt", "fold"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Public artifact frame is missing columns: {missing}")
    # A shallow view avoids duplicating the multi-million-row public feature table.
    # Existing numeric columns are treated as immutable; derived columns are new arrays.
    result = frame.copy(deep=False)
    base_models = np.asarray(base_model_oof, dtype=np.float64)
    if base_models.ndim != 2 or len(base_models) != len(result):
        raise ValueError("base_model_oof does not align with public artifact rows")
    result["base_model_mean"] = base_models.mean(axis=1).astype(np.float32)
    result["base_model_std"] = base_models.std(axis=1).astype(np.float32)
    result["base_model_range"] = (base_models.max(axis=1) - base_models.min(axis=1)).astype(np.float32)
    result["base_prediction_delta"] = (
        np.asarray(base_prediction, dtype=np.float64)
        - result["last_known_tvt"].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    result["y_true"] = (
        result["last_known_tvt"].to_numpy(dtype=np.float64)
        + result["target"].to_numpy(dtype=np.float64)
    )
    result["base_y_pred"] = np.asarray(base_prediction, dtype=np.float64)
    result["residual_target"] = result["y_true"] - result["base_y_pred"]

    desired = requested_columns or DEFAULT_PUBLIC_FEATURES
    columns = [name for name in desired if name in result.columns]
    derived = ["base_model_mean", "base_model_std", "base_model_range", "base_prediction_delta"]
    columns.extend(name for name in derived if name not in columns)
    if len(columns) < 10:
        raise ValueError(
            f"Only {len(columns)} residual features were found. The public artifact schema is incompatible."
        )
    for name in columns:
        if not pd.api.types.is_numeric_dtype(result[name]):
            result[name] = pd.to_numeric(result[name], errors="coerce")
        values = result[name].to_numpy(copy=False)
        if np.isinf(values).any():
            result[name] = result[name].replace([np.inf, -np.inf], np.nan)
    sampled = _sample_by_well(result, max_rows_per_well)
    return PublicResidualFeatures(frame=result, columns=columns, sampled=sampled)


def make_public_residual_model(config: dict[str, Any], seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=str(config.get("loss", "squared_error")),
        learning_rate=float(config.get("learning_rate", 0.035)),
        max_iter=int(config.get("max_iter", 220)),
        max_leaf_nodes=int(config.get("max_leaf_nodes", 15)),
        min_samples_leaf=int(config.get("min_samples_leaf", 50)),
        l2_regularization=float(config.get("l2_regularization", 15.0)),
        max_bins=int(config.get("max_bins", 127)),
        random_state=seed,
    )


def crossfit_public_residual(
    features: PublicResidualFeatures,
    fold_values: np.ndarray,
    model_config: dict[str, Any],
    model_seeds: list[int],
    target_clip: float,
) -> np.ndarray:
    frame = features.frame
    fold_array = np.asarray(fold_values)
    if len(fold_array) != len(frame):
        raise ValueError("fold_values do not align with residual features")
    output = np.empty(len(frame), dtype=np.float64)
    all_folds = sorted(int(value) for value in np.unique(fold_array))
    for fold in all_folds:
        valid = fold_array == fold
        sampled = features.sampled[fold_array[features.sampled] != fold]
        train = frame.iloc[sampled]
        x_train = train[features.columns].to_numpy(dtype=np.float32)
        y_train = np.clip(
            train["residual_target"].to_numpy(dtype=np.float64), -target_clip, target_clip
        )
        well_full = frame.groupby("well_id", sort=False).size()
        well_sampled = train.groupby("well_id", sort=False).size()
        weights = train["well_id"].map(well_full / well_sampled).to_numpy(dtype=np.float64)
        x_valid = frame.loc[valid, features.columns].to_numpy(dtype=np.float32)
        fold_predictions: list[np.ndarray] = []
        for seed in model_seeds:
            model = make_public_residual_model(model_config, seed + fold)
            model.fit(x_train, y_train, sample_weight=weights)
            fold_predictions.append(model.predict(x_valid))
        output[valid] = np.vstack(fold_predictions).mean(axis=0)
    return output


def fit_full_public_residual_models(
    features: PublicResidualFeatures,
    model_config: dict[str, Any],
    model_seeds: list[int],
    target_clip: float,
) -> list[HistGradientBoostingRegressor]:
    frame = features.frame
    train = frame.iloc[features.sampled]
    x = train[features.columns].to_numpy(dtype=np.float32)
    y = np.clip(train["residual_target"].to_numpy(dtype=np.float64), -target_clip, target_clip)
    well_full = frame.groupby("well_id", sort=False).size()
    well_sampled = train.groupby("well_id", sort=False).size()
    weights = train["well_id"].map(well_full / well_sampled).to_numpy(dtype=np.float64)
    models: list[HistGradientBoostingRegressor] = []
    for seed in model_seeds:
        model = make_public_residual_model(model_config, seed + 10_000)
        model.fit(x, y, sample_weight=weights)
        models.append(model)
    return models
