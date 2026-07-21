from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PhysicsSpec:
    degree: int
    tail: int | None

    @property
    def name(self) -> str:
        return f"poly_u_deg{self.degree}_{'all' if self.tail is None else f'tail{self.tail}'}"


def robust_poly_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    degree: int,
    iterations: int = 5,
) -> np.ndarray:
    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    xp = np.asarray(x_predict, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < degree + 2:
        return np.full(len(xp), float(np.nanmedian(y)) if len(y) else 0.0)
    origin = float(x[0])
    scale = max(float(np.ptp(x)), 1e-6)
    xs = (x - origin) / scale
    xps = (xp - origin) / scale
    coefficients = np.polyfit(xs, y, degree)
    for _ in range(iterations):
        residual = y - np.polyval(coefficients, xs)
        center = float(np.median(residual))
        spread = 1.4826 * float(np.median(np.abs(residual - center))) + 1e-6
        weights = 1.0 / (1.0 + np.square((residual - center) / (2.5 * spread)))
        coefficients = np.polyfit(xs, y, degree, w=weights)
    return np.polyval(coefficients, xps).astype(float)


def _predict_spec(
    md: np.ndarray,
    z: np.ndarray,
    tvt_input: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    spec: PhysicsSpec,
) -> np.ndarray:
    selected = train_indices
    if spec.tail is not None:
        selected = selected[-min(spec.tail, len(selected)) :]
    surface = tvt_input[selected] + z[selected]
    predicted_surface = robust_poly_predict(
        md[selected], surface, md[predict_indices], spec.degree
    )
    return predicted_surface - z[predict_indices]


def select_cutback_physics(
    horizontal: pd.DataFrame,
    *,
    cut_fractions: list[float],
    degrees: list[int],
    tails: list[int | None],
    minimum_holdout_rows: int = 35,
) -> tuple[np.ndarray, dict[str, Any]]:
    required = {"MD", "Z", "TVT_input"}
    missing = sorted(required - set(horizontal.columns))
    if missing:
        raise ValueError(f"horizontal frame is missing columns: {missing}")
    md = horizontal["MD"].to_numpy(dtype=float)
    z = horizontal["Z"].to_numpy(dtype=float)
    tvt_input = horizontal["TVT_input"].to_numpy(dtype=float)
    known = np.flatnonzero(np.isfinite(tvt_input))
    hidden = np.flatnonzero(~np.isfinite(tvt_input))
    if len(known) < 100 or len(hidden) == 0:
        return np.full(len(hidden), np.nan), {"status": "skip_short_prefix"}

    specs = [PhysicsSpec(degree, tail) for degree in degrees for tail in tails]
    scores: dict[str, list[float]] = {spec.name: [] for spec in specs}
    winners: list[str] = []
    default_scores: list[float] = []
    cut_rows: list[dict[str, Any]] = []
    for fraction in cut_fractions:
        cut = int(round(len(known) * float(fraction)))
        cut = max(50, min(cut, len(known) - minimum_holdout_rows))
        train_indices = known[:cut]
        holdout = known[cut:]
        if len(holdout) < minimum_holdout_rows:
            continue
        truth = tvt_input[holdout]
        default_rmse = float(np.sqrt(np.mean(np.square(truth - tvt_input[train_indices[-1]]))))
        default_scores.append(default_rmse)
        local: list[tuple[float, str]] = []
        for spec in specs:
            prediction = _predict_spec(md, z, tvt_input, train_indices, holdout, spec)
            rmse = float(np.sqrt(np.mean(np.square(prediction - truth))))
            scores[spec.name].append(rmse)
            local.append((rmse, spec.name))
        local.sort()
        winners.append(local[0][1])
        cut_rows.append(
            {
                "cut_fraction": float(fraction),
                "holdout_rows": int(len(holdout)),
                "best_name": local[0][1],
                "best_rmse": local[0][0],
                "anchor_rmse": default_rmse,
            }
        )
    if not winners:
        return np.full(len(hidden), np.nan), {"status": "skip_no_cutbacks"}

    aggregate = {
        name: float(np.median(values) + 0.10 * np.std(values))
        for name, values in scores.items()
        if values
    }
    best_name = min(aggregate, key=aggregate.get)
    best_spec = next(spec for spec in specs if spec.name == best_name)
    candidate = _predict_spec(md, z, tvt_input, known, hidden, best_spec)
    anchor_score = float(np.median(default_scores))
    consistency = float(sum(name == best_name for name in winners) / len(winners))
    return candidate, {
        "status": "ok",
        "best_name": best_name,
        "best_score": aggregate[best_name],
        "anchor_score": anchor_score,
        "anchor_gain": anchor_score - aggregate[best_name],
        "consistency": consistency,
        "known_rows": int(len(known)),
        "hidden_rows": int(len(hidden)),
        "cut_rows": cut_rows,
    }


def apply_cutback_profile(
    base: np.ndarray,
    physics: np.ndarray,
    md_since: np.ndarray,
    report: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    base_values = np.asarray(base, dtype=float)
    physics_values = np.asarray(physics, dtype=float)
    output = base_values.copy()
    status = str(report.get("status", "skip"))
    if status != "ok" or not np.isfinite(physics_values).all():
        return output, {"applied": False, "reason": status}
    difference = physics_values - base_values
    p95 = float(np.quantile(np.abs(difference), 0.95))
    eligible = (
        float(report["anchor_gain"]) >= float(profile.get("minimum_anchor_gain", 0.0))
        and float(report["best_score"]) <= float(profile.get("maximum_best_score", np.inf))
        and float(report["consistency"]) >= float(profile.get("minimum_consistency", 0.0))
        and p95 <= float(profile.get("maximum_difference_p95", np.inf))
    )
    if not eligible:
        return output, {"applied": False, "reason": "profile_gate", "difference_p95": p95}
    gain = max(float(report["anchor_gain"]), 0.0)
    alpha = min(
        float(profile.get("maximum_alpha", 1.0)),
        float(profile.get("base_alpha", 0.0)) + float(profile.get("gain_alpha", 0.0)) * gain,
    )
    tau = float(profile.get("fade_tau", 0.0))
    ramp = 1.0 - np.exp(-np.maximum(np.asarray(md_since, dtype=float), 0.0) / tau) if tau > 0 else 1.0
    cap = float(profile.get("correction_cap", np.inf))
    move = np.clip(alpha * ramp * difference, -cap, cap)
    output += move
    return output, {
        "applied": True,
        "alpha": alpha,
        "difference_p95": p95,
        "mean_abs_move": float(np.mean(np.abs(move))),
        "max_abs_move": float(np.max(np.abs(move))),
    }
