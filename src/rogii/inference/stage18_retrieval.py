from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


FEATURE_COLUMNS = [
    "donor_rank", "minimum_xyz_distance_ft", "median_xyz_distance_ft", "matched_prefix_points",
    "mean_abs_gr_difference", "typewell_gr_mean_difference", "requested_fraction",
    "prefix_rows_log", "suffix_rows_log", "prefix_u_rmse", "prefix_u_mae", "prefix_u_max_abs",
    "prefix_u_correlation", "calibration_offset_abs", "prefix_distance_mean", "prefix_distance_p90",
    "prefix_gr_delta_mean", "prefix_gr_delta_p90", "suffix_distance_mean", "suffix_distance_p90",
    "suffix_distance_max", "suffix_gr_delta_mean", "suffix_gr_delta_p90", "suffix_donor_u_std",
    "suffix_donor_u_change", "suffix_vs_base_rmse", "suffix_vs_base_mae",
]


class PortableRanker:
    """Minimal numeric HistGradientBoosting predictor independent of sklearn's pickle format."""

    def __init__(self, payload: dict[str, Any]):
        self.baseline = float(payload["baseline"])
        self.feature_columns = list(payload["feature_columns"])
        self.trees = payload["trees"]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature_columns].to_numpy(float)
        output = np.full(len(values), self.baseline, dtype=float)
        for tree in self.trees:
            feature = tree["feature_idx"]
            threshold = tree["num_threshold"]
            missing_left = tree["missing_go_to_left"]
            left, right = tree["left"], tree["right"]
            is_leaf, leaf_value = tree["is_leaf"], tree["value"]
            for row_index, row in enumerate(values):
                node = 0
                while not is_leaf[node]:
                    value = row[feature[node]]
                    go_left = bool(missing_left[node]) if np.isnan(value) else value <= threshold[node]
                    node = left[node] if go_left else right[node]
                output[row_index] += leaf_value[node]
        return output


def export_hist_gradient_boosting(model: Any) -> dict[str, Any]:
    trees = []
    for iteration in model._predictors:
        predictor = iteration[0]
        nodes = predictor.nodes
        if np.any(nodes["is_categorical"]):
            raise ValueError("Portable Stage 18 ranker does not support categorical splits")
        trees.append({
            "value": nodes["value"].astype(float).tolist(),
            "feature_idx": nodes["feature_idx"].astype(int).tolist(),
            "num_threshold": nodes["num_threshold"].astype(float).tolist(),
            "missing_go_to_left": nodes["missing_go_to_left"].astype(int).tolist(),
            "left": nodes["left"].astype(int).tolist(), "right": nodes["right"].astype(int).tolist(),
            "is_leaf": nodes["is_leaf"].astype(int).tolist(),
        })
    return {
        "format": "portable_hgb_numeric_v1", "baseline": float(np.asarray(model._baseline_prediction).reshape(-1)[0]),
        "feature_columns": FEATURE_COLUMNS, "trees": trees,
    }


def _sample_indices(length: int, count: int, stop: int | None = None) -> np.ndarray:
    stop = length if stop is None else min(int(stop), length)
    if stop <= 0:
        return np.empty(0, np.int64)
    return np.unique(np.linspace(0, stop - 1, min(count, stop)).round().astype(np.int64))


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3 or np.std(left[finite]) < 1e-8 or np.std(right[finite]) < 1e-8:
        return 0.0
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def _assigned_fold(well_id: str, edges: pd.DataFrame, assignments: pd.DataFrame, config: dict[str, Any]) -> tuple[int, str]:
    if well_id in assignments.index:
        return int(assignments.loc[well_id, "branch_group_fold"]), "frozen_training_assignment"
    close = edges[
        (edges["minimum_xyz_distance_ft"] <= float(config.get("branch_distance_ft", 150.0)))
        & (edges["matched_prefix_points"] >= int(config.get("minimum_matched_points", 3)))
    ].copy()
    if len(close):
        close["fold"] = close["donor_well_id"].map(assignments["branch_group_fold"]).astype(int)
        votes = close.groupby("fold", sort=True).agg(
            donors=("donor_well_id", "nunique"), minimum_distance=("minimum_xyz_distance_ft", "min")
        ).reset_index().sort_values(["donors", "minimum_distance", "fold"], ascending=[False, True, True], kind="stable")
        return int(votes.iloc[0]["fold"]), "visible_prefix_branch_vote"
    digest = hashlib.sha256(("stage18-test-fold:" + well_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % int(config.get("n_folds", 5)), "stable_hash_fallback"


def _feature_record(
    edge: pd.Series, requested_fraction: float, cut_index: int, target_u: np.ndarray,
    donor_u: np.ndarray, distance: np.ndarray, gr_delta: np.ndarray, base_suffix: np.ndarray,
    z: np.ndarray, prefix_rows: int,
) -> tuple[dict[str, float], np.ndarray, float]:
    tail_start = max(0, cut_index - prefix_rows)
    prefix_target, prefix_donor = target_u[tail_start:cut_index], donor_u[tail_start:cut_index]
    offset = float(np.median(prefix_target - prefix_donor))
    calibrated = donor_u + offset
    prefix_error = calibrated[tail_start:cut_index] - prefix_target
    prediction = calibrated[cut_index:] - z[cut_index:]
    pdist, pgr = distance[tail_start:cut_index], gr_delta[tail_start:cut_index]
    sdist, sgr = distance[cut_index:], gr_delta[cut_index:]
    record = {
        "donor_rank": float(edge["donor_rank"]), "minimum_xyz_distance_ft": float(edge["minimum_xyz_distance_ft"]),
        "median_xyz_distance_ft": float(edge["median_xyz_distance_ft"]),
        "matched_prefix_points": float(edge["matched_prefix_points"]),
        "mean_abs_gr_difference": float(edge["mean_abs_gr_difference"]),
        "typewell_gr_mean_difference": float(edge["typewell_gr_mean_difference"]),
        "requested_fraction": float(requested_fraction), "prefix_rows_log": float(np.log1p(cut_index)),
        "suffix_rows_log": float(np.log1p(len(prediction))),
        "prefix_u_rmse": float(np.sqrt(np.mean(np.square(prefix_error)))),
        "prefix_u_mae": float(np.mean(np.abs(prefix_error))), "prefix_u_max_abs": float(np.max(np.abs(prefix_error))),
        "prefix_u_correlation": _safe_correlation(calibrated[tail_start:cut_index], prefix_target),
        "calibration_offset_abs": abs(offset), "prefix_distance_mean": float(np.mean(pdist)),
        "prefix_distance_p90": float(np.quantile(pdist, 0.9)), "prefix_gr_delta_mean": float(np.mean(pgr)),
        "prefix_gr_delta_p90": float(np.quantile(pgr, 0.9)), "suffix_distance_mean": float(np.mean(sdist)),
        "suffix_distance_p90": float(np.quantile(sdist, 0.9)), "suffix_distance_max": float(np.max(sdist)),
        "suffix_gr_delta_mean": float(np.mean(sgr)), "suffix_gr_delta_p90": float(np.quantile(sgr, 0.9)),
        "suffix_donor_u_std": float(np.std(calibrated[cut_index:])),
        "suffix_donor_u_change": float(calibrated[-1] - calibrated[cut_index]),
        "suffix_vs_base_rmse": float(np.sqrt(np.mean(np.square(prediction - base_suffix)))),
        "suffix_vs_base_mae": float(np.mean(np.abs(prediction - base_suffix))),
    }
    return record, prediction, offset


def _weighted_path(
    predictions: np.ndarray, distances: np.ndarray, gr_delta: np.ndarray,
    distance_scale: float, gr_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.exp(-np.minimum(distances, 5000.0) / distance_scale)
    weights *= np.exp(-np.minimum(gr_delta, 500.0) / gr_scale)
    total = weights.sum(axis=0)
    path = np.divide((weights * predictions).sum(axis=0), total, out=np.full(predictions.shape[1], np.nan), where=total > 1e-12)
    return path, total


def apply_ranked_retrieval(package_dir: str | Path, data_dir: str | Path, submission_path: str | Path) -> dict[str, Any]:
    package, data, submission_path = Path(package_dir), Path(data_dir), Path(submission_path)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    config = dict(manifest["inference"])
    assignments = pd.read_parquet(package / "well_assignments.parquet").set_index("well_id")
    models = {}
    for fold in range(int(config.get("n_folds", 5))):
        payload = json.loads((package / f"ranker_fold_{fold}.json").read_text(encoding="utf-8"))
        models[fold] = PortableRanker(payload)

    sample = pd.read_csv(data / "sample_submission.csv")
    submission = pd.read_csv(submission_path)
    if list(submission.columns) != ["id", "tvt"] or not submission["id"].astype(str).equals(sample["id"].astype(str)):
        raise AssertionError("Stage 18 input submission does not match sample order")
    split = submission["id"].astype(str).str.rsplit("_", n=1, expand=True)
    submission["well_id"], submission["row_index"] = split[0], split[1].astype(int)

    train_files = sorted((data / "train").glob("*__horizontal_well.csv"))
    train_wells = [path.name.split("__", 1)[0] for path in train_files]
    train_frames: dict[str, pd.DataFrame] = {}
    points, typewell_mean = [], {}
    trajectory_samples = int(config.get("trajectory_samples", 48))
    for position, (well_id, path) in enumerate(zip(train_wells, train_files, strict=True), 1):
        frame = pd.read_csv(path, usecols=["X", "Y", "Z", "GR", "TVT"])
        train_frames[well_id] = frame
        take = _sample_indices(len(frame), trajectory_samples)
        sampled = frame.iloc[take][["X", "Y", "Z", "GR"]].copy()
        sampled["well_id"] = well_id
        points.append(sampled)
        tw = pd.read_csv(data / "train" / f"{well_id}__typewell.csv", usecols=["GR"])
        typewell_mean[well_id] = float(pd.to_numeric(tw["GR"], errors="coerce").mean())
        if position % 100 == 0:
            print(f"Stage18 loaded {position}/{len(train_files)} donor wells", flush=True)
    point_table = pd.concat(points, ignore_index=True)
    point_xyz = point_table[["X", "Y", "Z"]].to_numpy(float)
    point_wells = point_table["well_id"].astype(str).to_numpy()
    point_gr = pd.to_numeric(point_table["GR"], errors="coerce").to_numpy(float)
    global_index = NearestNeighbors(
        n_neighbors=min(int(config.get("point_neighbors", 64)), len(point_table)), algorithm="kd_tree"
    ).fit(point_xyz)

    audit_rows, output = [], submission.copy()
    for well_id, group in submission.groupby("well_id", sort=True):
        horizontal = pd.read_csv(data / "test" / f"{well_id}__horizontal_well.csv")
        known = pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
        cut_index = int(known.sum())
        if not np.array_equal(known, np.arange(len(horizontal)) < cut_index):
            raise AssertionError(f"{well_id}: known TVT_input is not a contiguous prefix")
        hidden_rows = group["row_index"].to_numpy(int)
        expected_rows = np.arange(cut_index, len(horizontal))
        if not np.array_equal(np.sort(hidden_rows), expected_rows):
            raise AssertionError(f"{well_id}: submission rows do not match hidden suffix")
        base_full = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
        row_to_value = dict(zip(group["row_index"].astype(int), group["tvt"].astype(float), strict=True))
        base_full[expected_rows] = [row_to_value[int(row)] for row in expected_rows]
        take = _sample_indices(len(horizontal), int(config.get("query_prefix_samples", 24)), cut_index)
        query = horizontal.iloc[take]
        query_xyz = query[["X", "Y", "Z"]].to_numpy(float)
        _, neighbor_index = global_index.kneighbors(query_xyz)
        distances = np.sqrt(np.sum((point_xyz[neighbor_index] - query_xyz[:, None, :]) ** 2, axis=2))
        query_gr = pd.to_numeric(query["GR"], errors="coerce").to_numpy(float)
        candidates: dict[str, dict[str, list[float]]] = {}
        for qpos in range(len(query)):
            seen: set[str] = set()
            for distance, pidx in zip(distances[qpos], neighbor_index[qpos], strict=True):
                donor_id = str(point_wells[pidx])
                # Competition test wells can share IDs with training wells. Never transfer the same well's hidden TVT.
                if donor_id == str(well_id) or donor_id in seen:
                    continue
                seen.add(donor_id)
                entry = candidates.setdefault(donor_id, {"distance": [], "gr_delta": []})
                entry["distance"].append(float(distance))
                if np.isfinite(query_gr[qpos]) and np.isfinite(point_gr[pidx]):
                    entry["gr_delta"].append(abs(float(query_gr[qpos] - point_gr[pidx])))
        ranked_candidates = sorted(
            candidates.items(), key=lambda item: (round(min(item[1]["distance"]), 8), -len(item[1]["distance"]), item[0])
        )[: int(config.get("donor_neighbors", 16))]
        target_tw = pd.read_csv(data / "test" / f"{well_id}__typewell.csv", usecols=["GR"])
        target_tw_mean = float(pd.to_numeric(target_tw["GR"], errors="coerce").mean())
        edges = pd.DataFrame.from_records([{
            "donor_well_id": donor_id, "donor_rank": rank,
            "minimum_xyz_distance_ft": min(values["distance"]),
            "median_xyz_distance_ft": float(np.median(values["distance"])),
            "matched_prefix_points": len(values["distance"]),
            "mean_abs_gr_difference": float(np.mean(values["gr_delta"])) if values["gr_delta"] else float("nan"),
            "typewell_gr_mean_difference": abs(target_tw_mean - typewell_mean[donor_id]),
        } for rank, (donor_id, values) in enumerate(ranked_candidates, 1)])
        fold, fold_source = _assigned_fold(str(well_id), edges, assignments, config)
        edges = edges[edges["donor_well_id"].map(assignments["branch_group_fold"]).astype(int) != fold].head(
            int(config.get("maximum_candidates", 12))
        )
        target_xyz = horizontal[["X", "Y", "Z"]].to_numpy(float)
        target_gr = pd.to_numeric(horizontal["GR"], errors="coerce").interpolate(limit_direction="both").to_numpy(float)
        target_u = base_full + horizontal["Z"].to_numpy(float)
        records, aligned = [], {}
        for edge in edges.itertuples(index=False):
            donor_id = str(edge.donor_well_id)
            donor = train_frames[donor_id]
            donor_index = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(donor[["X", "Y", "Z"]].to_numpy(float))
            distance, nearest = donor_index.kneighbors(target_xyz)
            nearest = nearest[:, 0]
            donor_u = donor["TVT"].to_numpy(float)[nearest] + donor["Z"].to_numpy(float)[nearest]
            donor_gr = pd.to_numeric(donor["GR"], errors="coerce").interpolate(limit_direction="both").to_numpy(float)[nearest]
            gr_delta = np.abs(target_gr - donor_gr)
            feature, donor_prediction, offset = _feature_record(
                pd.Series(edge._asdict()), cut_index / len(horizontal), cut_index, target_u, donor_u,
                distance[:, 0], gr_delta, base_full[cut_index:], horizontal["Z"].to_numpy(float),
                int(config.get("prefix_calibration_rows", 256)),
            )
            records.append({"donor_well_id": donor_id, **feature})
            aligned[donor_id] = (donor_prediction, distance[:, 0][cut_index:], gr_delta[cut_index:], offset)
        feature_frame = pd.DataFrame.from_records(records)
        if len(feature_frame) < int(config.get("minimum_donors", 2)):
            audit_rows.append({"well_id": well_id, "status": "kept_base_insufficient_donors", "assigned_fold": fold})
            continue
        feature_frame["score"] = models[fold].predict(feature_frame[FEATURE_COLUMNS])
        chosen = feature_frame.sort_values(["score", "donor_well_id"], kind="stable").head(
            int(config.get("selected_donors", 4))
        )["donor_well_id"].astype(str).tolist()
        donor_predictions = np.stack([aligned[donor][0] for donor in chosen])
        donor_distances = np.stack([aligned[donor][1] for donor in chosen])
        donor_gr_delta = np.stack([aligned[donor][2] for donor in chosen])
        retrieved, confidence = _weighted_path(
            donor_predictions, donor_distances, donor_gr_delta,
            float(config.get("distance_scale_ft", 300.0)), float(config.get("gr_scale", 30.0)),
        )
        active = np.isfinite(retrieved) & (confidence > float(config.get("minimum_weight", 1e-5)))
        corrected = base_full[cut_index:].copy()
        blend = float(config.get("blend_weight", 0.20))
        corrected[active] += blend * (retrieved[active] - corrected[active])
        for row, value in zip(expected_rows, corrected, strict=True):
            output.loc[(output["well_id"] == well_id) & (output["row_index"] == int(row)), "tvt"] = float(value)
        audit_rows.append({
            "well_id": well_id, "status": "applied", "assigned_fold": fold, "fold_source": fold_source,
            "candidate_donors": int(len(feature_frame)), "selected_donors": chosen,
            "active_fraction": float(active.mean()),
            "mean_abs_move": float(np.mean(np.abs(corrected - base_full[cut_index:]))),
            "max_abs_move": float(np.max(np.abs(corrected - base_full[cut_index:]))),
        })

    final = output[["id", "tvt"]].copy()
    if not final["id"].astype(str).equals(sample["id"].astype(str)) or not np.isfinite(final["tvt"]).all():
        raise AssertionError("Stage 18 output audit failed")
    final.to_csv(submission_path, index=False, lineterminator="\n")
    audit = {
        "stage18_retrieval_applied": True, "rows": len(final), "wells": len(audit_rows),
        "submission_sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(), "well_report": audit_rows,
    }
    (submission_path.parent / "stage18_retrieval_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("STAGE18_RETRIEVAL_AUDIT =", audit, flush=True)
    return audit
