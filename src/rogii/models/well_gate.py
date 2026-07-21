from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


WELL_GATE_FEATURES = [
    "known_rows",
    "hidden_rows",
    "hidden_to_known_ratio",
    "best_score",
    "anchor_score",
    "anchor_gain",
    "consistency",
    "degree",
    "tail",
    "base_std",
    "base_range",
    "base_slope",
    "physics_base_mean",
    "physics_base_std",
    "physics_base_rmse",
    "physics_base_p95",
    "physics_base_max",
    "physics_base_slope",
    "conservative_move_mean",
    "conservative_move_rmse",
    "conservative_move_max",
    "md_span",
]


def _spec_numbers(name: object) -> tuple[float, float]:
    match = re.fullmatch(r"poly_u_deg(\d+)_(?:tail(\d+)|all)", str(name))
    if match is None:
        return 0.0, 0.0
    degree = float(match.group(1))
    tail = float(match.group(2)) if match.group(2) is not None else 1_000_000.0
    return degree, tail


def build_well_gate_table(
    candidate_matrix: pd.DataFrame,
    well_reports: pd.DataFrame,
) -> pd.DataFrame:
    forbidden = {"y_true", "target", "actual_gain", "gain_target"}
    records: list[dict[str, float | str | int]] = []
    reports = well_reports.copy()
    reports["well_id"] = reports["well_id"].astype(str)
    report_map = reports.set_index("well_id").to_dict("index")
    for well_id, part in candidate_matrix.groupby("well_id", sort=False):
        well = str(well_id)
        report = report_map.get(well, {})
        base = part["base_y_pred"].to_numpy(dtype=float)
        physics = part["physics_y_pred"].to_numpy(dtype=float)
        conservative = part["pred_conservative"].to_numpy(dtype=float)
        md_column = "evaluation_md" if "evaluation_md" in part else "MD"
        md = part[md_column].to_numpy(dtype=float)
        difference = physics - base
        move = conservative - base
        valid = np.isfinite(difference)
        degree, tail = _spec_numbers(report.get("best_name"))
        slope = lambda values: float(np.polyfit(md, values, 1)[0]) if len(values) >= 2 and np.ptp(md) > 0 else 0.0
        record: dict[str, float | str | int] = {
            "well_id": well,
            "fold": int(part["fold"].iloc[0]),
            "spatial_fold": int(part["spatial_fold"].iloc[0]),
            "n_rows": int(len(part)),
            "known_rows": float(report.get("known_rows", 0.0)),
            "hidden_rows": float(report.get("hidden_rows", len(part))),
            "best_score": float(report.get("best_score", np.nan)),
            "anchor_score": float(report.get("anchor_score", np.nan)),
            "anchor_gain": float(report.get("anchor_gain", np.nan)),
            "consistency": float(report.get("consistency", 0.0)),
            "degree": degree,
            "tail": tail,
            "base_std": float(np.std(base)),
            "base_range": float(np.ptp(base)),
            "base_slope": slope(base),
            "physics_base_mean": float(np.mean(difference[valid])) if valid.any() else np.nan,
            "physics_base_std": float(np.std(difference[valid])) if valid.any() else np.nan,
            "physics_base_rmse": float(np.sqrt(np.mean(np.square(difference[valid])))) if valid.any() else np.nan,
            "physics_base_p95": float(np.quantile(np.abs(difference[valid]), 0.95)) if valid.any() else np.nan,
            "physics_base_max": float(np.max(np.abs(difference[valid]))) if valid.any() else np.nan,
            "physics_base_slope": slope(np.nan_to_num(difference, nan=0.0)),
            "conservative_move_mean": float(np.mean(move)),
            "conservative_move_rmse": float(np.sqrt(np.mean(np.square(move)))),
            "conservative_move_max": float(np.max(np.abs(move))),
            "md_span": float(np.ptp(md)),
        }
        record["hidden_to_known_ratio"] = float(record["hidden_rows"]) / max(float(record["known_rows"]), 1.0)
        truth = part["y_true"].to_numpy(dtype=float)
        record["base_rmse"] = float(np.sqrt(np.mean(np.square(base - truth))))
        record["conservative_rmse"] = float(np.sqrt(np.mean(np.square(conservative - truth))))
        record["gain_target"] = float(record["base_rmse"] - record["conservative_rmse"])
        records.append(record)
    result = pd.DataFrame.from_records(records)
    if forbidden.intersection(WELL_GATE_FEATURES):
        raise AssertionError("Target-derived column entered WELL_GATE_FEATURES")
    return result


def make_well_gate_model(config: dict[str, Any], seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=float(config.get("learning_rate", 0.04)),
        max_iter=int(config.get("max_iter", 140)),
        max_leaf_nodes=int(config.get("max_leaf_nodes", 7)),
        min_samples_leaf=int(config.get("min_samples_leaf", 30)),
        l2_regularization=float(config.get("l2_regularization", 12.0)),
        max_bins=int(config.get("max_bins", 127)),
        random_state=int(seed),
    )


def fit_well_gate(
    table: pd.DataFrame,
    train_mask: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> HistGradientBoostingRegressor:
    model = make_well_gate_model(config, seed)
    train = table.loc[train_mask]
    target = np.clip(
        train["gain_target"].to_numpy(dtype=float),
        -float(config.get("target_clip", 10.0)),
        float(config.get("target_clip", 10.0)),
    )
    model.fit(
        train[WELL_GATE_FEATURES].to_numpy(dtype=np.float32),
        target,
        sample_weight=np.sqrt(train["n_rows"].to_numpy(dtype=float)),
    )
    return model


def predict_well_gate(model: HistGradientBoostingRegressor, table: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        model.predict(table[WELL_GATE_FEATURES].to_numpy(dtype=np.float32)), dtype=float
    )
