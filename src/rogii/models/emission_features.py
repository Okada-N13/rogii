from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rogii.models.raw_ncc import alignment_costs


COST_CHANNELS = ("ncc_w5", "ncc_w13", "ncc_w25", "ncc_mix")
CANDIDATE_CHANNELS = (*COST_CHANNELS, "horizontal_gr_z", "candidate_typewell_gr_z", "candidate_valid")


@dataclass
class EmissionSequence:
    cut_id: str
    well_id: str
    fold: int
    spatial_fold: int
    typewell_fold: int
    cut_fraction: float
    row_index: np.ndarray
    md: np.ndarray
    costs: np.ndarray
    row_features: np.ndarray
    target_state: np.ndarray
    valid: np.ndarray
    true_offset: np.ndarray
    surface_y_pred: np.ndarray
    y_true: np.ndarray


def _evenly_spaced(length: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, maximum).round().astype(np.int64))


def build_emission_sequence(
    record: pd.Series | Any,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
    *,
    weight: float,
    correction_cap_ft: float,
) -> EmissionSequence:
    positions, surface, offsets, cost_map = alignment_costs(
        record,
        horizontal,
        typewell,
        config,
        weight=weight,
        correction_cap_ft=correction_cap_ft,
    )
    missing = [name for name in COST_CHANNELS if name not in cost_map]
    if missing:
        raise KeyError(f"Missing Stage 12B NCC channels: {missing}")
    selected = _evenly_spaced(len(positions), int(config.get("max_rows_per_cut", 256)))
    positions = positions[selected]
    surface = surface[selected].astype(np.float32)
    ncc_costs = np.stack([cost_map[name][selected] for name in COST_CHANNELS], axis=1)
    ncc_costs = np.nan_to_num(ncc_costs, nan=3.0, posinf=3.0, neginf=0.0).clip(0.0, 3.0)

    suffix = horizontal.iloc[positions]
    md = pd.to_numeric(suffix["MD"], errors="coerce").to_numpy(dtype=float)
    gr = pd.to_numeric(suffix["GR"], errors="coerce").to_numpy(dtype=float)
    truth = pd.to_numeric(suffix["TVT"], errors="coerce").to_numpy(dtype=float)
    true_offset = truth - surface
    target = np.argmin(np.abs(offsets[None, :] - true_offset[:, None]), axis=1)
    in_grid = np.isfinite(true_offset) & (true_offset >= offsets[0]) & (true_offset <= offsets[-1])
    row = np.arange(len(target))
    true_costs = ncc_costs[row, :, target]
    valid = in_grid & np.any(true_costs < 2.999, axis=1)

    horizon = np.maximum(md - float(record.anchor_md), 0.0)
    horizon_scale = max(float(np.nanmax(horizon)), 1.0)
    finite_gr = gr[np.isfinite(gr)]
    center = float(np.median(finite_gr)) if len(finite_gr) else 0.0
    spread = float(np.median(np.abs(finite_gr - center)) * 1.4826) if len(finite_gr) else 1.0
    spread = max(spread, 1.0)
    gr_z = np.nan_to_num((gr - center) / spread, nan=0.0).clip(-6.0, 6.0)
    type_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(dtype=float)
    type_gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(dtype=float)
    finite_type = np.isfinite(type_tvt) & np.isfinite(type_gr)
    type_tvt, type_gr = type_tvt[finite_type], type_gr[finite_type]
    order = np.argsort(type_tvt)
    type_tvt, type_gr = type_tvt[order], type_gr[order]
    candidate_tvt = surface[:, None] + offsets[None, :]
    candidate_gr = np.interp(candidate_tvt.ravel(), type_tvt, type_gr).reshape(candidate_tvt.shape)
    candidate_gr_z = np.clip((candidate_gr - center) / spread, -6.0, 6.0)
    candidate_valid = ((candidate_tvt >= type_tvt[0]) & (candidate_tvt <= type_tvt[-1])).astype(float)
    horizontal_gr_z = np.broadcast_to(gr_z[:, None], candidate_gr_z.shape)
    paired = np.stack([horizontal_gr_z, candidate_gr_z, candidate_valid], axis=1)
    costs = np.concatenate([ncc_costs, paired], axis=1).astype(np.float32)
    slope = np.gradient(surface.astype(float), md) if len(md) > 1 else np.zeros(len(md))
    row_features = np.column_stack(
        [
            horizon / horizon_scale,
            np.full(len(md), float(record.cut_fraction)),
            np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0).clip(-2.0, 2.0),
            gr_z,
        ]
    ).astype(np.float32)
    return EmissionSequence(
        cut_id=str(record.cut_id),
        well_id=str(record.well_id),
        fold=int(record.fold),
        spatial_fold=int(record.spatial_fold),
        typewell_fold=int(record.typewell_fold),
        cut_fraction=float(record.cut_fraction),
        row_index=suffix["row_index"].to_numpy(dtype=np.int64),
        md=md.astype(np.float32),
        costs=costs.astype(np.float16),
        row_features=row_features,
        target_state=target.astype(np.int16),
        valid=valid,
        true_offset=true_offset.astype(np.float32),
        surface_y_pred=surface,
        y_true=truth.astype(np.float32),
    )


def feature_invariance(
    record: pd.Series | Any,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
    *,
    weight: float,
    correction_cap_ft: float,
) -> bool:
    original = build_emission_sequence(
        record, horizontal, typewell, config, weight=weight, correction_cap_ft=correction_cap_ft
    )
    changed = horizontal.copy()
    changed.loc[changed.index[int(record.cut_index):], "TVT"] += 997.0
    perturbed = build_emission_sequence(
        record, changed, typewell, config, weight=weight, correction_cap_ft=correction_cap_ft
    )
    return bool(
        np.array_equal(original.row_index, perturbed.row_index)
        and np.array_equal(original.costs, perturbed.costs)
        and np.array_equal(original.row_features, perturbed.row_features)
        and np.array_equal(original.surface_y_pred, perturbed.surface_y_pred)
    )
