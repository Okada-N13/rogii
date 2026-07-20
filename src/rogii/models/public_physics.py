from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PhysicsSpec:
    delta_scale: float = 1.0
    poly_column: str | None = None
    physics_weight: float = 0.0
    correction_cap: float = 0.0
    fade_tau: float = 85.0

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


def robust_poly_predict(
    x_known: np.ndarray,
    y_known: np.ndarray,
    x_target: np.ndarray,
    degree: int,
    iterations: int = 5,
) -> np.ndarray:
    x = np.asarray(x_known, dtype=np.float64)
    y = np.asarray(y_known, dtype=np.float64)
    target = np.asarray(x_target, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < max(12, degree + 2):
        return np.full(len(target), np.nan, dtype=np.float64)
    origin = float(x[-1])
    scale = max(float(np.ptp(x)), 1.0)
    xs = (x - origin) / scale
    xt = (target - origin) / scale
    degree = min(int(degree), len(x) - 1)
    coefficients = np.polyfit(xs, y, degree)
    for _ in range(iterations):
        residual = y - np.polyval(coefficients, xs)
        center = float(np.median(residual))
        robust_scale = max(1.4826 * float(np.median(np.abs(residual - center))), 1e-6)
        weights = 1.0 / (1.0 + np.square(residual / (2.5 * robust_scale)))
        coefficients = np.polyfit(xs, y, degree, w=weights)
    return np.polyval(coefficients, xt)


def build_prefix_physics_features(
    base: pd.DataFrame,
    data_dir: Path,
    tails: list[int],
    degrees: list[int],
) -> pd.DataFrame:
    required = {"well_id", "row_index", "y_pred", "Z", "MD"}
    missing = sorted(required - set(base.columns))
    if missing:
        raise ValueError(f"Base OOF is missing columns: {missing}")
    result = pd.DataFrame(index=base.index)
    result["anchor_tvt"] = np.nan
    result["base_delta"] = np.nan
    poly_names = [f"poly_u_deg{degree}_tail{tail}" for tail in tails for degree in degrees]
    for name in poly_names:
        result[name] = np.nan

    grouped = base.groupby("well_id", sort=False).groups
    for number, (well_id, indices) in enumerate(grouped.items(), 1):
        positions = np.asarray(indices, dtype=np.int64)
        rows = base.loc[positions, "row_index"].to_numpy(dtype=np.int64)
        path = data_dir / "train" / f"{well_id}__horizontal_well.csv"
        horizontal = pd.read_csv(path, usecols=["MD", "Z", "TVT_input"])
        first_hidden = int(rows.min())
        known = horizontal.iloc[:first_hidden].dropna(subset=["MD", "Z", "TVT_input"])
        if len(known) < 30:
            raise ValueError(f"{well_id}: only {len(known)} visible-prefix rows")
        anchor = float(known["TVT_input"].iloc[-1])
        target_md = horizontal.iloc[rows]["MD"].to_numpy(dtype=np.float64)
        target_z = horizontal.iloc[rows]["Z"].to_numpy(dtype=np.float64)
        result.loc[positions, "anchor_tvt"] = anchor
        result.loc[positions, "base_delta"] = (
            base.loc[positions, "y_pred"].to_numpy(dtype=np.float64) - anchor
        )
        known_md = known["MD"].to_numpy(dtype=np.float64)
        known_surface = (
            known["TVT_input"].to_numpy(dtype=np.float64)
            + known["Z"].to_numpy(dtype=np.float64)
        )
        for tail in tails:
            selected = slice(max(0, len(known) - int(tail)), len(known))
            for degree in degrees:
                name = f"poly_u_deg{degree}_tail{tail}"
                surface = robust_poly_predict(
                    known_md[selected], known_surface[selected], target_md, degree
                )
                poly_tvt = surface - target_z
                result.loc[positions, name] = (
                    poly_tvt - base.loc[positions, "y_pred"].to_numpy(dtype=np.float64)
                )
        if number % 100 == 0:
            print(f"prefix physics: {number}/{len(grouped)} wells", flush=True)
    if result.isna().any().any():
        bad = result.columns[result.isna().any()].tolist()
        raise ValueError(f"Prefix physics produced missing values in: {bad}")
    return result.astype(np.float32)


def make_physics_specs(config: dict[str, object], poly_columns: list[str]) -> list[PhysicsSpec]:
    delta_scales = [float(value) for value in config.get("delta_scales", [0.98, 1.0, 1.02])]
    weights = [float(value) for value in config.get("physics_weights", [0.03, 0.06, 0.10])]
    caps = [float(value) for value in config.get("correction_caps", [6.0, 12.0])]
    taus = [float(value) for value in config.get("fade_taus", [85.0, 160.0])]
    specs = [PhysicsSpec(delta_scale=scale) for scale in delta_scales]
    for scale in delta_scales:
        for column in poly_columns:
            for weight in weights:
                for cap in caps:
                    for tau in taus:
                        specs.append(PhysicsSpec(scale, column, weight, cap, tau))
    return specs


def apply_physics_spec(
    base: pd.DataFrame,
    features: pd.DataFrame,
    spec: PhysicsSpec,
) -> np.ndarray:
    prediction = base["y_pred"].to_numpy(dtype=np.float64).copy()
    prediction += (spec.delta_scale - 1.0) * features["base_delta"].to_numpy(dtype=np.float64)
    if spec.poly_column is not None and spec.physics_weight != 0.0:
        difference = features[spec.poly_column].to_numpy(dtype=np.float64)
        ramp = 1.0 - np.exp(-np.maximum(base["MD"].to_numpy(dtype=np.float64), 0.0) / spec.fade_tau)
        prediction += spec.physics_weight * ramp * np.clip(
            difference, -spec.correction_cap, spec.correction_cap
        )
    return prediction


def nested_select_predictions(
    base: pd.DataFrame,
    features: pd.DataFrame,
    folds: np.ndarray,
    specs: list[PhysicsSpec],
    minimum_selection_gain: float,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    fold_array = np.asarray(folds)
    truth = base["y_true"].to_numpy(dtype=np.float64)
    base_prediction = base["y_pred"].to_numpy(dtype=np.float64)
    base_squared = np.square(base_prediction - truth)
    unique_folds = np.asarray(sorted(np.unique(fold_array)))
    fold_masks = [fold_array == fold for fold in unique_folds]
    fold_counts = np.asarray([mask.sum() for mask in fold_masks], dtype=np.int64)
    spec_fold_sse = np.empty((len(specs), len(unique_folds)), dtype=np.float64)
    # Retain only fold-level sufficient statistics. Keeping every 3.8M-row
    # candidate prediction would exceed standard Colab memory.
    for index, spec in enumerate(specs):
        prediction = apply_physics_spec(base, features, spec)
        squared = np.square(prediction - truth)
        spec_fold_sse[index] = [float(squared[mask].sum()) for mask in fold_masks]
        if (index + 1) % 25 == 0:
            print(f"physics grid: {index + 1}/{len(specs)} candidates", flush=True)
    base_fold_sse = np.asarray([float(base_squared[mask].sum()) for mask in fold_masks])
    output = base_prediction.copy()
    selections: list[dict[str, object]] = []
    total_count = int(fold_counts.sum())
    for fold_position, fold in enumerate(unique_folds):
        valid = fold_masks[fold_position]
        train_count = total_count - int(fold_counts[fold_position])
        base_train_sse = float(base_fold_sse.sum() - base_fold_sse[fold_position])
        base_rmse = float(np.sqrt(base_train_sse / train_count))
        train_sse = spec_fold_sse.sum(axis=1) - spec_fold_sse[:, fold_position]
        scores = np.sqrt(train_sse / train_count)
        best_index = int(np.argmin(scores))
        best_gain = base_rmse - float(scores[best_index])
        if best_gain >= minimum_selection_gain:
            selected: PhysicsSpec | None = specs[best_index]
            selected_prediction = apply_physics_spec(base, features, selected)
            output[valid] = selected_prediction[valid]
        else:
            best_index = -1
            selected = None
        selections.append(
            {
                "fold": int(fold),
                "selection_base_rmse": base_rmse,
                "selection_best_rmse": float(scores.min()),
                "selection_gain": best_gain,
                "selected_index": best_index,
                "selected_spec": selected.to_dict() if selected else None,
            }
        )
    ranking: list[dict[str, object]] = []
    base_rmse = float(np.sqrt(base_squared.mean()))
    for index, spec in enumerate(specs):
        rmse = float(np.sqrt(spec_fold_sse[index].sum() / total_count))
        ranking.append(
            {"index": index, "rmse": rmse, "rmse_delta": rmse - base_rmse, **spec.to_dict()}
        )
    ranking.sort(key=lambda row: float(row["rmse"]))
    return output, selections, ranking
