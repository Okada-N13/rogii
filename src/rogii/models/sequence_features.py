from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SEQUENCE_FEATURE_NAMES = [
    "base_delta",
    "md_since",
    "eval_fraction",
    "z_delta",
    "dz_dmd",
    "x_delta",
    "y_delta",
    "dx_dmd",
    "dy_dmd",
    "gr_normalized",
    "gr_roll9",
    "gr_roll31",
    "gr_gradient",
    "typewell_gr_normalized",
    "gr_typewell_residual",
    "typewell_gr_gradient",
    "base_surface_delta",
    "base_gradient",
    "base_curvature",
    "physics_base_delta",
    "conservative_move",
]


@dataclass(frozen=True)
class SequenceWell:
    well_id: str
    ids: np.ndarray
    row_index: np.ndarray
    md: np.ndarray
    features: np.ndarray
    residual_target: np.ndarray
    base_prediction: np.ndarray
    y_true: np.ndarray
    fold: int


def _safe_gradient(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(len(values), dtype=float)
    delta = np.gradient(coordinate)
    delta = np.where(np.abs(delta) < 1e-6, 1.0, delta)
    return np.gradient(values) / delta


def build_sequence_well(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    prediction_rows: pd.DataFrame,
) -> SequenceWell:
    required = {
        "id",
        "well_id",
        "row_index",
        "fold",
        "y_true",
        "base_y_pred",
        "physics_y_pred",
        "pred_conservative",
    }
    missing = sorted(required - set(prediction_rows.columns))
    if missing:
        raise ValueError(f"prediction rows are missing columns: {missing}")
    allowed_horizontal = horizontal[["MD", "Z", "X", "Y", "GR", "TVT_input"]].copy()
    rows = prediction_rows.sort_values("row_index").copy()
    index = rows["row_index"].to_numpy(dtype=int)
    if len(index) == 0 or index.min() < 0 or index.max() >= len(allowed_horizontal):
        raise ValueError("prediction row indices do not align with the horizontal well")
    known = allowed_horizontal[allowed_horizontal["TVT_input"].notna()]
    if known.empty:
        raise ValueError("horizontal well has no visible TVT prefix")
    anchor = known.iloc[-1]
    last_tvt = float(anchor["TVT_input"])
    last_md = float(anchor["MD"])
    last_z = float(anchor["Z"])

    md = allowed_horizontal.loc[index, "MD"].to_numpy(dtype=float)
    z = allowed_horizontal.loc[index, "Z"].to_numpy(dtype=float)
    x = allowed_horizontal.loc[index, "X"].to_numpy(dtype=float)
    y = allowed_horizontal.loc[index, "Y"].to_numpy(dtype=float)
    gr_all = allowed_horizontal["GR"].astype(float).interpolate(limit_direction="both")
    known_gr = gr_all.loc[known.index].to_numpy(dtype=float)
    gr_center = float(np.nanmedian(known_gr))
    gr_scale = 1.4826 * float(np.nanmedian(np.abs(known_gr - gr_center)))
    if not np.isfinite(gr_scale) or gr_scale < 1.0:
        gr_scale = max(float(np.nanstd(known_gr)), 10.0)
    gr_series = (gr_all - gr_center) / gr_scale
    gr = gr_series.loc[index].to_numpy(dtype=float)
    gr_roll9 = gr_series.rolling(9, center=True, min_periods=1).mean().loc[index].to_numpy(dtype=float)
    gr_roll31 = gr_series.rolling(31, center=True, min_periods=1).mean().loc[index].to_numpy(dtype=float)

    tw = typewell[["TVT", "GR"]].dropna().sort_values("TVT")
    tw = tw.groupby("TVT", as_index=False)["GR"].mean()
    tw_tvt = tw["TVT"].to_numpy(dtype=float)
    tw_gr = tw["GR"].to_numpy(dtype=float)
    base = rows["base_y_pred"].to_numpy(dtype=float)
    expected_gr = np.interp(base, tw_tvt, tw_gr)
    tw_gradient = _safe_gradient(tw_gr, tw_tvt)
    expected_gradient = np.interp(base, tw_tvt, tw_gradient)

    md_since = md - last_md
    denominator = max(float(md_since[-1]), 1.0)
    base_gradient = _safe_gradient(base, md)
    physics = rows["physics_y_pred"].to_numpy(dtype=float)
    conservative = rows["pred_conservative"].to_numpy(dtype=float)
    features = np.column_stack(
        [
            base - last_tvt,
            md_since,
            md_since / denominator,
            z - last_z,
            _safe_gradient(z, md),
            x - float(anchor["X"]),
            y - float(anchor["Y"]),
            _safe_gradient(x, md),
            _safe_gradient(y, md),
            gr,
            gr_roll9,
            gr_roll31,
            _safe_gradient(gr, md),
            (expected_gr - gr_center) / gr_scale,
            (gr_all.loc[index].to_numpy(dtype=float) - expected_gr) / gr_scale,
            expected_gradient,
            (base + z) - (last_tvt + last_z),
            base_gradient,
            _safe_gradient(base_gradient, md),
            physics - base,
            conservative - base,
        ]
    ).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    y_true = rows["y_true"].to_numpy(dtype=float)
    return SequenceWell(
        well_id=str(rows["well_id"].iloc[0]),
        ids=rows["id"].astype(str).to_numpy(),
        row_index=index.astype(np.int32),
        md=md.astype(np.float32),
        features=features,
        residual_target=(y_true - base).astype(np.float32),
        base_prediction=base.astype(np.float32),
        y_true=y_true.astype(np.float32),
        fold=int(rows["fold"].iloc[0]),
    )


def feature_standardizer(wells: list[SequenceWell]) -> tuple[np.ndarray, np.ndarray]:
    if not wells:
        raise ValueError("Cannot standardize an empty well list")
    count = 0
    total = np.zeros(len(SEQUENCE_FEATURE_NAMES), dtype=np.float64)
    squared = np.zeros(len(SEQUENCE_FEATURE_NAMES), dtype=np.float64)
    for well in wells:
        values = well.features.astype(np.float64)
        count += len(values)
        total += values.sum(axis=0)
        squared += np.square(values).sum(axis=0)
    mean = total / count
    variance = np.maximum(squared / count - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
