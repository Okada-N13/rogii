from __future__ import annotations

import numpy as np
import pandas as pd


def apply_alignment_profile(
    frame: pd.DataFrame,
    diagnostics: pd.DataFrame,
    profile: dict[str, object],
) -> tuple[np.ndarray, dict[str, float | int]]:
    branch = str(profile.get("branch", "loose"))
    weight = float(profile.get("weight", 0.2))
    correction = frame[f"correction_{branch}"].to_numpy(dtype=float).copy()
    well_ids = frame["well_id"].astype(str)
    report = diagnostics[diagnostics["branch"].astype(str) == branch].copy()
    report["well_id"] = report["well_id"].astype(str)
    report = report.set_index("well_id")
    active = report.get("active", pd.Series(1.0, index=report.index)).reindex(well_ids).fillna(0.0).to_numpy() > 0
    prefix = report["prefix_shape_corr"].reindex(well_ids).fillna(-np.inf).to_numpy(dtype=float)
    cost_gain = report["cost_gain_per_step"].reindex(well_ids).fillna(-np.inf).to_numpy(dtype=float)
    keep = active
    minimum_prefix = profile.get("minimum_prefix_correlation")
    if minimum_prefix is not None:
        keep &= prefix >= float(minimum_prefix)
    minimum_cost = profile.get("minimum_cost_gain")
    if minimum_cost is not None:
        keep &= cost_gain >= float(minimum_cost)
    maximum = profile.get("correction_cap")
    if maximum is not None:
        correction = np.clip(correction, -float(maximum), float(maximum))
    correction[~keep] = 0.0
    prediction = frame["base_y_pred"].to_numpy(dtype=float) + weight * correction
    selected_wells = int(pd.Series(well_ids[keep]).nunique())
    return prediction, {
        "selected_wells": selected_wells,
        "rejected_wells": int(well_ids.nunique() - selected_wells),
        "mean_abs_move": float(np.mean(np.abs(weight * correction))),
        "max_abs_move": float(np.max(np.abs(weight * correction))),
    }
