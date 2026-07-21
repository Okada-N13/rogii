from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


TARGET_COLUMNS = ("target_slope_correction", "target_curvature")


def make_cut_indices(
    n_rows: int,
    fractions: Sequence[float],
    min_prefix_rows: int = 64,
    min_suffix_rows: int = 64,
) -> list[int]:
    if n_rows < min_prefix_rows + min_suffix_rows:
        return []
    lower = int(min_prefix_rows)
    upper = int(n_rows - min_suffix_rows)
    cuts = {
        int(np.clip(round(float(fraction) * n_rows), lower, upper))
        for fraction in fractions
        if 0.0 < float(fraction) < 1.0
    }
    return sorted(cuts)


def _finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _summary(prefix: str, values: Iterable[Any]) -> dict[str, float]:
    array = _finite(values)
    if len(array) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_q10": 0.0,
            f"{prefix}_q50": 0.0,
            f"{prefix}_q90": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array)),
        f"{prefix}_q10": float(np.quantile(array, 0.10)),
        f"{prefix}_q50": float(np.quantile(array, 0.50)),
        f"{prefix}_q90": float(np.quantile(array, 0.90)),
    }


def _robust_slope(md: np.ndarray, values: np.ndarray, window_ft: float) -> float:
    finite = np.isfinite(md) & np.isfinite(values)
    md = md[finite]
    values = values[finite]
    if len(md) < 3:
        return 0.0
    use = md >= float(md[-1] - window_ft)
    if int(use.sum()) < 3:
        use = np.ones(len(md), dtype=bool)
    x = md[use] - float(md[use][-1])
    y = values[use] - float(values[use][-1])
    coefficient = float(np.polyfit(x, y, 1)[0])
    residual = y - coefficient * x
    scale = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 1e-6)
    keep = np.abs(residual - np.median(residual)) <= 3.0 * scale
    if int(keep.sum()) >= 3:
        coefficient = float(np.polyfit(x[keep], y[keep], 1)[0])
    return coefficient


def typewell_signature(typewell: pd.DataFrame) -> dict[str, float]:
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(dtype=float)
    gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(dtype=float)
    finite_tvt = tvt[np.isfinite(tvt)]
    signature = _summary("typewell_gr", gr)
    signature.update(
        {
            "typewell_tvt_min": float(np.min(finite_tvt)) if len(finite_tvt) else 0.0,
            "typewell_tvt_max": float(np.max(finite_tvt)) if len(finite_tvt) else 0.0,
            "typewell_tvt_span": float(np.ptp(finite_tvt)) if len(finite_tvt) else 0.0,
            "typewell_rows": int(len(typewell)),
        }
    )
    return signature


def build_cut_record(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cut_index: int,
    *,
    prefix_window_ft: float = 800.0,
    target_ridge: float = 1e-3,
) -> dict[str, float | int | str]:
    required = {"well_id", "row_index", "MD", "X", "Y", "Z", "GR", "TVT"}
    missing = sorted(required - set(horizontal.columns))
    if missing:
        raise ValueError(f"horizontal frame is missing columns {missing}")
    if not 3 <= int(cut_index) < len(horizontal) - 2:
        raise ValueError("cut_index must leave at least three prefix and suffix rows")

    cut_index = int(cut_index)
    prefix = horizontal.iloc[:cut_index]
    suffix = horizontal.iloc[cut_index:]
    anchor = prefix.iloc[-1]
    md_prefix = prefix["MD"].to_numpy(dtype=float)
    z_prefix = prefix["Z"].to_numpy(dtype=float)
    tvt_prefix = prefix["TVT"].to_numpy(dtype=float)
    u_prefix = tvt_prefix + z_prefix
    anchor_md = float(anchor["MD"])
    anchor_u = float(anchor["TVT"] + anchor["Z"])
    prefix_u_slope = _robust_slope(md_prefix, u_prefix, prefix_window_ft) * 1000.0
    prefix_tvt_slope = _robust_slope(md_prefix, tvt_prefix, prefix_window_ft) * 1000.0

    suffix_md = suffix["MD"].to_numpy(dtype=float)
    horizon_kft = (suffix_md - anchor_md) / 1000.0
    true_u_delta = (
        suffix["TVT"].to_numpy(dtype=float)
        + suffix["Z"].to_numpy(dtype=float)
        - anchor_u
    )
    target_residual = true_u_delta - prefix_u_slope * horizon_kft
    design = np.column_stack([horizon_kft, np.square(horizon_kft)])
    penalty = np.eye(2, dtype=float) * float(target_ridge)
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target_residual)

    end = horizontal.iloc[-1]
    dx = float(end["X"] - anchor["X"])
    dy = float(end["Y"] - anchor["Y"])
    dz = float(end["Z"] - anchor["Z"])
    horizon_md = float(end["MD"] - anchor_md)
    full_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(dtype=float)
    prefix_gr = full_gr[:cut_index]
    suffix_gr = full_gr[cut_index:]
    record: dict[str, float | int | str] = {
        "well_id": str(anchor["well_id"]),
        "cut_id": f"{anchor['well_id']}__cut{cut_index}",
        "cut_index": cut_index,
        "n_rows": int(len(horizontal)),
        "cut_fraction": float(cut_index / len(horizontal)),
        "suffix_rows": int(len(suffix)),
        "anchor_md": anchor_md,
        "anchor_x": float(anchor["X"]),
        "anchor_y": float(anchor["Y"]),
        "anchor_z": float(anchor["Z"]),
        "anchor_u": anchor_u,
        "anchor_tvt": float(anchor["TVT"]),
        "prefix_u_slope_per_kft": prefix_u_slope,
        "prefix_tvt_slope_per_kft": prefix_tvt_slope,
        "horizon_md_ft": horizon_md,
        "horizon_dx_ft": dx,
        "horizon_dy_ft": dy,
        "horizon_dz_ft": dz,
        "horizon_xy_ft": float(np.hypot(dx, dy)),
        "horizon_xyz_ft": float(np.sqrt(dx * dx + dy * dy + dz * dz)),
        "trajectory_dx_per_md": dx / max(horizon_md, 1.0),
        "trajectory_dy_per_md": dy / max(horizon_md, 1.0),
        "trajectory_dz_per_md": dz / max(horizon_md, 1.0),
        "gr_missing_fraction": float(np.mean(~np.isfinite(full_gr))),
        "gr_prefix_suffix_mean_delta": float(np.nanmean(suffix_gr) - np.nanmean(prefix_gr)),
        "target_slope_correction": float(coefficients[0]),
        "target_curvature": float(coefficients[1]),
    }
    record.update(_summary("prefix_gr", prefix_gr))
    record.update(_summary("suffix_gr", suffix_gr))
    record.update(_summary("full_gr", full_gr))
    record.update(typewell_signature(typewell))
    return record


def build_multicut_records(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    cuts = make_cut_indices(
        len(horizontal),
        fractions=config.get("fractions", [0.35, 0.50, 0.65, 0.80]),
        min_prefix_rows=int(config.get("min_prefix_rows", 64)),
        min_suffix_rows=int(config.get("min_suffix_rows", 64)),
    )
    records = [
        build_cut_record(
            horizontal,
            typewell,
            cut,
            prefix_window_ft=float(config.get("prefix_window_ft", 800.0)),
            target_ridge=float(config.get("target_ridge", 1e-3)),
        )
        for cut in cuts
    ]
    return pd.DataFrame.from_records(records)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "well_id",
        "cut_id",
        "target_slope_correction",
        "target_curvature",
        "fold",
        "spatial_fold",
        "typewell_fold",
    }
    return sorted(
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    )


def feature_schema_hash(columns: Sequence[str]) -> str:
    payload = json.dumps(list(columns), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
