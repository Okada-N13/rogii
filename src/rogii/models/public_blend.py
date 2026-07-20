from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BlendSpec:
    branch: str
    weight: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def align_package_ground_truth(
    base: pd.DataFrame,
    package_ground_truth: pd.DataFrame,
    target_tolerance: float = 0.05,
) -> tuple[np.ndarray, dict[str, object]]:
    required = {"id", "last_known_TVT", "target_tvt"}
    missing = sorted(required - set(package_ground_truth.columns))
    if missing:
        raise ValueError(f"Package ground truth is missing columns: {missing}")
    package_ids = package_ground_truth["id"].astype(str)
    base_ids = base["id"].astype(str)
    if len(package_ids) != len(base_ids):
        raise ValueError(f"Package rows {len(package_ids)} != base rows {len(base_ids)}")
    order_matches = package_ids.reset_index(drop=True).equals(base_ids.reset_index(drop=True))
    if order_matches:
        order = np.arange(len(base), dtype=np.int64)
    else:
        if package_ids.duplicated().any() or base_ids.duplicated().any():
            raise ValueError("Cannot align duplicated IDs")
        position = pd.Series(np.arange(len(package_ids), dtype=np.int64), index=package_ids)
        order = position.reindex(base_ids).to_numpy()
        if pd.isna(order).any():
            raise ValueError("Package and base IDs do not contain the same rows")
        order = order.astype(np.int64)
    package_truth = package_ground_truth["target_tvt"].to_numpy(dtype=np.float64)[order]
    base_truth = base["y_true"].to_numpy(dtype=np.float64)
    target_difference = package_truth - base_truth
    max_abs = float(np.max(np.abs(target_difference)))
    rmse = float(np.sqrt(np.mean(np.square(target_difference))))
    if max_abs > target_tolerance:
        raise ValueError(
            f"Package target does not align with base target: max_abs={max_abs:.6f}"
        )
    return order, {
        "rows": len(base),
        "id_order_matches": bool(order_matches),
        "target_max_abs_difference": max_abs,
        "target_rmse_difference": rmse,
    }


def load_package_branches(
    package_root: str,
    package_ground_truth: pd.DataFrame,
    order: np.ndarray,
    branch_files: dict[str, str],
) -> dict[str, np.ndarray]:
    from pathlib import Path

    last = package_ground_truth["last_known_TVT"].to_numpy(dtype=np.float32)[order]
    branches: dict[str, np.ndarray] = {}
    for name, relative in branch_files.items():
        path = Path(package_root) / relative
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 1 or len(values) != len(order):
            raise ValueError(f"{path}: expected one-dimensional {len(order)}-row OOF")
        delta = np.asarray(values[order], dtype=np.float32)
        if not np.isfinite(delta).all():
            raise ValueError(f"{path}: OOF contains non-finite values")
        branches[name] = last + delta
    return branches


def make_blend_specs(branches: list[str], weights: list[float]) -> list[BlendSpec]:
    return [BlendSpec(branch, float(weight)) for branch in branches for weight in weights]


def apply_blend_spec(
    base_prediction: np.ndarray,
    branches: dict[str, np.ndarray],
    spec: BlendSpec,
) -> np.ndarray:
    base = np.asarray(base_prediction, dtype=np.float64)
    branch = np.asarray(branches[spec.branch], dtype=np.float64)
    return base + spec.weight * (branch - base)


def nested_select_blend(
    base: pd.DataFrame,
    branches: dict[str, np.ndarray],
    folds: np.ndarray,
    specs: list[BlendSpec],
    minimum_selection_gain: float,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    fold_array = np.asarray(folds)
    truth = base["y_true"].to_numpy(dtype=np.float64)
    base_prediction = base["y_pred"].to_numpy(dtype=np.float64)
    unique_folds = np.asarray(sorted(np.unique(fold_array)))
    fold_masks = [fold_array == fold for fold in unique_folds]
    fold_counts = np.asarray([mask.sum() for mask in fold_masks], dtype=np.int64)
    base_squared = np.square(base_prediction - truth)
    base_fold_sse = np.asarray([float(base_squared[mask].sum()) for mask in fold_masks])
    spec_fold_sse = np.empty((len(specs), len(unique_folds)), dtype=np.float64)
    for index, spec in enumerate(specs):
        prediction = apply_blend_spec(base_prediction, branches, spec)
        squared = np.square(prediction - truth)
        spec_fold_sse[index] = [float(squared[mask].sum()) for mask in fold_masks]
    output = base_prediction.copy()
    total_count = int(fold_counts.sum())
    selections: list[dict[str, object]] = []
    for position, fold in enumerate(unique_folds):
        train_count = total_count - int(fold_counts[position])
        base_train_rmse = float(
            np.sqrt((base_fold_sse.sum() - base_fold_sse[position]) / train_count)
        )
        train_sse = spec_fold_sse.sum(axis=1) - spec_fold_sse[:, position]
        scores = np.sqrt(train_sse / train_count)
        best_index = int(np.argmin(scores))
        gain = base_train_rmse - float(scores[best_index])
        selected = specs[best_index] if gain >= minimum_selection_gain else None
        if selected is not None:
            valid = fold_masks[position]
            prediction = apply_blend_spec(base_prediction, branches, selected)
            output[valid] = prediction[valid]
        selections.append(
            {
                "fold": int(fold),
                "selection_base_rmse": base_train_rmse,
                "selection_best_rmse": float(scores.min()),
                "selection_gain": gain,
                "selected_index": best_index if selected else -1,
                "selected_spec": selected.to_dict() if selected else None,
            }
        )
    base_rmse = float(np.sqrt(base_fold_sse.sum() / total_count))
    ranking = []
    for index, spec in enumerate(specs):
        rmse = float(np.sqrt(spec_fold_sse[index].sum() / total_count))
        ranking.append(
            {"index": index, "rmse": rmse, "rmse_delta": rmse - base_rmse, **spec.to_dict()}
        )
    ranking.sort(key=lambda row: float(row["rmse"]))
    return output, selections, ranking

