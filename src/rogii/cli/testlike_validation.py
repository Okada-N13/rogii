from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.data.loading import discover_horizontal_wells, load_horizontal_well, load_typewell, well_id_from_path
from rogii.data.multicut import build_cut_record, feature_columns, make_cut_indices, typewell_signature


CONTROL_NAMES = ("last_tvt", "constant_u", "linear_u")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 16B test-like pseudo-test validation builder")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit-wells", type=int)
    return parser


def _sample_indices(length: int, count: int, stop: int | None = None) -> np.ndarray:
    stop = length if stop is None else min(int(stop), length)
    if stop <= 0:
        return np.empty(0, np.int64)
    return np.unique(np.linspace(0, stop - 1, min(count, stop)).round().astype(np.int64))


def _frame_hash(frame: pd.DataFrame) -> str:
    """Canonical cross-version hash; do not depend on pandas' internal hash keys."""
    digest = hashlib.sha256()
    for column in frame.columns:
        digest.update(str(column).encode("utf-8") + b"\0")
        values = frame[column]
        if pd.api.types.is_bool_dtype(values):
            array = values.to_numpy(np.uint8, copy=True)
            digest.update(b"bool\0" + array.tobytes())
        elif pd.api.types.is_integer_dtype(values):
            array = values.to_numpy("<i8", copy=True)
            digest.update(b"int64\0" + array.tobytes())
        elif pd.api.types.is_numeric_dtype(values):
            array = values.to_numpy("<f8", copy=True)
            array[np.isnan(array)] = np.nan
            digest.update(b"float64\0" + array.tobytes())
        else:
            digest.update(b"string\0")
            for value in values:
                encoded = ("<NA>" if pd.isna(value) else str(value)).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little") + encoded)
    return digest.hexdigest()


def _slope(md: np.ndarray, values: np.ndarray, window_ft: float) -> float:
    finite = np.isfinite(md) & np.isfinite(values)
    md, values = md[finite], values[finite]
    if len(md) < 3:
        return 0.0
    use = md >= md[-1] - float(window_ft)
    if use.sum() < 3:
        use[:] = True
    return float(np.polyfit(md[use] - md[use][-1], values[use] - values[use][-1], 1)[0])


def _well_summary(horizontal: pd.DataFrame, typewell: pd.DataFrame, samples: int) -> tuple[dict[str, Any], pd.DataFrame]:
    index = _sample_indices(len(horizontal), samples)
    sampled = horizontal.iloc[index]
    signature = typewell_signature(typewell)
    record: dict[str, Any] = {
        "well_id": str(horizontal["well_id"].iloc[0]), "n_rows": len(horizontal),
        "x": float(horizontal["X"].median()), "y": float(horizontal["Y"].median()),
        "z": float(horizontal["Z"].median()),
        "start_x": float(horizontal["X"].iloc[0]), "start_y": float(horizontal["Y"].iloc[0]),
        "end_x": float(horizontal["X"].iloc[-1]), "end_y": float(horizontal["Y"].iloc[-1]),
        **signature,
    }
    points = sampled[["well_id", "row_index", "X", "Y", "Z", "GR"]].copy()
    return record, points


def _pseudo_manifest(horizontal_by_well: dict[str, pd.DataFrame], config: dict[str, Any]) -> pd.DataFrame:
    primary = [float(value) for value in config.get("fractions", [])]
    diagnostic = [float(value) for value in config.get("diagnostic_fractions", [])]
    rows: list[dict[str, Any]] = []
    visible_columns = ["MD", "X", "Y", "Z", "GR", "TVT"]
    for well_id, horizontal in horizontal_by_well.items():
        labels = [(value, "primary") for value in primary] + [(value, "diagnostic") for value in diagnostic]
        for requested, role in labels:
            cuts = make_cut_indices(
                len(horizontal), [requested], int(config.get("min_prefix_rows", 64)),
                int(config.get("min_suffix_rows", 64)),
            )
            if not cuts:
                continue
            cut = cuts[0]
            visible = horizontal.iloc[:cut][visible_columns].copy()
            rows.append({
                "well_id": well_id, "cut_id": f"{well_id}__cut{cut}", "cut_index": cut,
                "n_rows": len(horizontal), "suffix_rows": len(horizontal) - cut,
                "requested_fraction": requested, "cut_fraction": cut / len(horizontal),
                "evaluation_role": role, "target_visible_to_features": False,
                "visible_prefix_sha256": _frame_hash(visible),
            })
    result = pd.DataFrame.from_records(rows)
    if result["cut_id"].duplicated().any():
        raise RuntimeError("Duplicate Stage 16B cut IDs")
    return result


def _donor_graph(
    manifest: pd.DataFrame,
    horizontal_by_well: dict[str, pd.DataFrame],
    point_table: pd.DataFrame,
    summaries: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    coordinates = point_table[["X", "Y", "Z"]].to_numpy(np.float32)
    point_wells = point_table["well_id"].astype(str).to_numpy()
    point_gr = pd.to_numeric(point_table["GR"], errors="coerce").to_numpy(float)
    neighbors = min(int(config.get("point_neighbors", 64)), len(point_table))
    index = NearestNeighbors(n_neighbors=neighbors, algorithm="kd_tree").fit(coordinates)
    signature = summaries.set_index("well_id")
    output: list[dict[str, Any]] = []
    for position, cut in enumerate(manifest.itertuples(index=False), 1):
        horizontal = horizontal_by_well[str(cut.well_id)]
        take = _sample_indices(len(horizontal), int(config.get("query_prefix_samples", 24)), int(cut.cut_index))
        query = horizontal.iloc[take]
        distance, neighbor_index = index.kneighbors(query[["X", "Y", "Z"]].to_numpy(np.float32))
        query_gr = pd.to_numeric(query["GR"], errors="coerce").to_numpy(float)
        candidates: dict[str, dict[str, list[float]]] = {}
        for qpos in range(len(query)):
            seen: set[str] = set()
            for dist, pidx in zip(distance[qpos], neighbor_index[qpos], strict=True):
                donor = str(point_wells[pidx])
                if donor == str(cut.well_id) or donor in seen:
                    continue
                seen.add(donor)
                entry = candidates.setdefault(donor, {"distance": [], "gr_delta": []})
                entry["distance"].append(float(dist))
                if np.isfinite(query_gr[qpos]) and np.isfinite(point_gr[pidx]):
                    entry["gr_delta"].append(abs(float(query_gr[qpos] - point_gr[pidx])))
        ranked = sorted(candidates.items(), key=lambda item: (min(item[1]["distance"]), -len(item[1]["distance"])))
        for rank, (donor, values) in enumerate(ranked[: int(config.get("donor_neighbors", 16))], 1):
            left, right = signature.loc[str(cut.well_id)], signature.loc[donor]
            output.append({
                "cut_id": str(cut.cut_id), "well_id": str(cut.well_id), "donor_well_id": donor,
                "donor_rank": rank, "minimum_xyz_distance_ft": min(values["distance"]),
                "median_xyz_distance_ft": float(np.median(values["distance"])),
                "matched_prefix_points": len(values["distance"]),
                "mean_abs_gr_difference": float(np.mean(values["gr_delta"])) if values["gr_delta"] else float("nan"),
                "typewell_gr_mean_difference": abs(float(left["typewell_gr_mean"] - right["typewell_gr_mean"])),
            })
        if position % 500 == 0:
            print(f"built donor graph for {position}/{len(manifest)} cuts", flush=True)
    return pd.DataFrame.from_records(output)


class _UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _branch_groups(wells: list[str], graph: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    union = _UnionFind(wells)
    close = graph[
        (graph["minimum_xyz_distance_ft"] <= float(config.get("branch_distance_ft", 150.0)))
        & (graph["matched_prefix_points"] >= int(config.get("minimum_matched_points", 3)))
    ]
    for row in close.itertuples(index=False):
        union.union(str(row.well_id), str(row.donor_well_id))
    roots = {well: union.find(well) for well in wells}
    unique = {root: index for index, root in enumerate(sorted(set(roots.values())))}
    return pd.DataFrame({"well_id": wells, "branch_group": [unique[roots[well]] for well in wells]})


def _balanced_group_folds(groups: pd.DataFrame, n_folds: int) -> pd.Series:
    sizes = groups.groupby("branch_group").size().sort_values(ascending=False)
    loads = [0] * n_folds
    assignment: dict[int, int] = {}
    for group, size in sizes.items():
        fold = int(np.argmin(loads))
        assignment[int(group)] = fold
        loads[fold] += int(size)
    return groups["branch_group"].map(assignment).astype(np.int16)


def _stable_standard_folds(wells: pd.Series, n_folds: int, seed: int) -> pd.Series:
    """Version-independent balanced assignment based only on well ID and seed."""
    keys = wells.astype(str).map(
        lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    )
    order = np.argsort(keys.to_numpy(), kind="stable")
    values = np.empty(len(wells), np.int16)
    values[order] = np.arange(len(wells), dtype=np.int64) % int(n_folds)
    return pd.Series(values, index=wells.index, name="fold")


def _stable_spatial_folds(wells: pd.DataFrame, n_folds: int) -> pd.Series:
    """Deterministic recursive median spatial partition without KMeans."""
    groups = [wells.index.to_numpy(np.int64)]
    while len(groups) < int(n_folds):
        split_at = max(range(len(groups)), key=lambda index: (len(groups[index]), -int(groups[index].min())))
        selected = groups.pop(split_at)
        values = wells.loc[selected, ["x", "y"]]
        spans = values.max() - values.min()
        dimension = "x" if float(spans["x"]) >= float(spans["y"]) else "y"
        ordered = wells.loc[selected].sort_values([dimension, "well_id"], kind="stable").index.to_numpy(np.int64)
        middle = len(ordered) // 2
        groups.extend([ordered[:middle], ordered[middle:]])
    centroids = [
        (float(wells.loc[group, "x"].mean()), float(wells.loc[group, "y"].mean()), group)
        for group in groups
    ]
    output = pd.Series(np.full(len(wells), -1, np.int16), index=wells.index, name="spatial_fold")
    for fold, (_, _, group) in enumerate(sorted(centroids, key=lambda value: (value[0], value[1]))):
        output.loc[group] = fold
    if (output < 0).any():
        raise RuntimeError("Stable spatial fold assignment is incomplete")
    return output


def _assignments(summaries: pd.DataFrame, groups: pd.DataFrame, config: dict[str, Any], seed: int) -> pd.DataFrame:
    result = summaries.merge(groups, on="well_id", validate="one_to_one")
    result["fold"] = _stable_standard_folds(
        result["well_id"], min(int(config.get("n_standard_folds", 5)), len(result)), seed
    )
    result["spatial_fold"] = _stable_spatial_folds(
        result, min(int(config.get("n_spatial_folds", 6)), len(result))
    )
    result["branch_group_fold"] = _balanced_group_folds(
        result[["well_id", "branch_group"]], min(int(config.get("n_branch_group_folds", 5)), len(result))
    )
    return result


def _control_metrics(
    manifest: pd.DataFrame,
    horizontal_by_well: dict[str, pd.DataFrame],
    assignments: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_map = assignments.set_index("well_id")[["fold", "spatial_fold", "branch_group_fold"]]
    rows: list[dict[str, Any]] = []
    window = float(config.get("prefix_slope_window_ft", 800.0))
    for cut in manifest.itertuples(index=False):
        frame = horizontal_by_well[str(cut.well_id)]
        prefix, suffix = frame.iloc[: int(cut.cut_index)], frame.iloc[int(cut.cut_index):]
        anchor = prefix.iloc[-1]
        md = suffix["MD"].to_numpy(float)
        z = suffix["Z"].to_numpy(float)
        truth = suffix["TVT"].to_numpy(float)
        prefix_md = prefix["MD"].to_numpy(float)
        prefix_u = prefix["TVT"].to_numpy(float) + prefix["Z"].to_numpy(float)
        u_slope = _slope(prefix_md, prefix_u, window)
        horizon = md - float(anchor["MD"])
        predictions = {
            "last_tvt": np.full(len(suffix), float(anchor["TVT"])),
            "constant_u": float(anchor["TVT"] + anchor["Z"]) - z,
            "linear_u": float(anchor["TVT"] + anchor["Z"]) + u_slope * horizon - z,
        }
        folds = fold_map.loc[str(cut.well_id)]
        for name, prediction in predictions.items():
            error = prediction - truth
            rows.append({
                "cut_id": str(cut.cut_id), "well_id": str(cut.well_id), "control": name,
                "evaluation_role": str(cut.evaluation_role), "requested_fraction": float(cut.requested_fraction),
                "cut_fraction": float(cut.cut_fraction), "n_rows": len(error),
                "sse": float(np.square(error).sum()), "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "bias": float(np.mean(error)), "fold": int(folds["fold"]),
                "spatial_fold": int(folds["spatial_fold"]), "branch_group_fold": int(folds["branch_group_fold"]),
            })
    metrics = pd.DataFrame.from_records(rows)
    report: dict[str, Any] = {}
    for name in CONTROL_NAMES:
        selected = metrics[metrics["control"] == name]
        total_rows, total_sse = int(selected["n_rows"].sum()), float(selected["sse"].sum())
        item: dict[str, Any] = {
            "pooled_rmse": float(np.sqrt(total_sse / total_rows)), "n_rows": total_rows,
            "n_cuts": len(selected), "cut_rmse_p90": float(selected["rmse"].quantile(0.9)),
            "cut_rmse_max": float(selected["rmse"].max()),
        }
        for role, frame in selected.groupby("evaluation_role"):
            item[f"{role}_rmse"] = float(np.sqrt(frame["sse"].sum() / frame["n_rows"].sum()))
        for fraction, frame in selected.groupby("requested_fraction"):
            item[f"fraction_{float(fraction):.2f}_rmse"] = float(np.sqrt(frame["sse"].sum() / frame["n_rows"].sum()))
        for family in ("fold", "spatial_fold", "branch_group_fold"):
            for fold, frame in selected.groupby(family):
                item[f"{family}_{int(fold)}_rmse"] = float(np.sqrt(frame["sse"].sum() / frame["n_rows"].sum()))
        report[name] = item
    return metrics, report


def _invariance(horizontal_by_well: dict[str, pd.DataFrame], typewell_by_well: dict[str, pd.DataFrame], config: dict[str, Any]) -> dict[str, Any]:
    checks = []
    fraction = float(config.get("fractions", [0.26])[len(config.get("fractions", [0.26])) // 2])
    for well_id in sorted(horizontal_by_well)[: min(8, len(horizontal_by_well))]:
        frame = horizontal_by_well[well_id]
        cuts = make_cut_indices(len(frame), [fraction], int(config.get("min_prefix_rows", 64)), int(config.get("min_suffix_rows", 64)))
        if not cuts:
            continue
        cut = cuts[0]
        original = build_cut_record(frame, typewell_by_well[well_id], cut)
        changed = frame.copy()
        changed.loc[changed.index[cut:], "TVT"] += 997.0
        perturbed = build_cut_record(changed, typewell_by_well[well_id], cut)
        columns = feature_columns(pd.DataFrame([original]))
        same = np.allclose([original[c] for c in columns], [perturbed[c] for c in columns], equal_nan=True)
        target_changed = any(not np.isclose(original[c], perturbed[c]) for c in ("target_slope_correction", "target_curvature"))
        checks.append({"well_id": well_id, "cut_index": cut, "features_invariant": bool(same), "target_changed": bool(target_changed)})
    return {"passed": bool(checks) and all(row["features_invariant"] and row["target_changed"] for row in checks), "checks": checks}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    pseudo, donor, validation = dict(config["pseudo_test"]), dict(config["donor_graph"]), dict(config["validation"])
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = discover_horizontal_wells(args.data_dir, "train")
    if args.limit_wells is not None:
        paths = paths[: args.limit_wells]
    horizontal_by_well, typewell_by_well = {}, {}
    summaries, points = [], []
    for index, path in enumerate(paths, 1):
        horizontal, typewell = load_horizontal_well(path), load_typewell(path)
        well_id = well_id_from_path(path)
        horizontal_by_well[well_id], typewell_by_well[well_id] = horizontal, typewell
        summary, sampled = _well_summary(horizontal, typewell, int(donor.get("trajectory_samples", 48)))
        summaries.append(summary); points.append(sampled)
        if index % 100 == 0:
            print(f"loaded {index}/{len(paths)} wells", flush=True)
    summary_frame = pd.DataFrame.from_records(summaries)
    point_table = pd.concat(points, ignore_index=True)
    manifest = _pseudo_manifest(horizontal_by_well, pseudo)
    graph = _donor_graph(manifest, horizontal_by_well, point_table, summary_frame, donor)
    groups = _branch_groups(sorted(horizontal_by_well), graph, donor)
    assignments = _assignments(summary_frame, groups, validation, int(config.get("seed", 42)))
    manifest = manifest.merge(assignments[["well_id", "fold", "spatial_fold", "branch_group", "branch_group_fold"]], on="well_id", validate="many_to_one")
    cut_metrics, control_report = _control_metrics(manifest, horizontal_by_well, assignments, pseudo)
    invariance = _invariance(horizontal_by_well, typewell_by_well, pseudo)
    if not invariance["passed"]:
        raise AssertionError(f"Stage 16B hidden target invariance failed: {invariance}")
    manifest.to_parquet(output / "pseudo_test_manifest.parquet", index=False)
    assignments.to_parquet(output / "well_assignments.parquet", index=False)
    graph.to_parquet(output / "donor_graph.parquet", index=False)
    groups.to_parquet(output / "branch_groups.parquet", index=False)
    cut_metrics.to_parquet(output / "control_cut_metrics.parquet", index=False)
    write_json(output / "control_metrics.json", control_report)
    write_json(output / "hidden_target_invariance.json", invariance)
    core_columns = [
        "well_id", "cut_id", "cut_index", "n_rows", "suffix_rows", "requested_fraction",
        "cut_fraction", "evaluation_role", "target_visible_to_features", "visible_prefix_sha256",
    ]
    fold_columns = ["well_id", "fold", "spatial_fold", "branch_group", "branch_group_fold"]
    donor_columns = ["cut_id", "well_id", "donor_well_id", "donor_rank"]
    summary = {
        "stage16b_complete": True, "n_wells": len(horizontal_by_well), "n_cuts": len(manifest),
        "n_primary_cuts": int((manifest["evaluation_role"] == "primary").sum()),
        "n_donor_edges": len(graph), "n_branch_groups": int(groups["branch_group"].nunique()),
        "hidden_target_invariance": True,
        "manifest_sha256": _frame_hash(manifest.sort_values("cut_id")[core_columns].reset_index(drop=True)),
        "fold_sha256": _frame_hash(assignments.sort_values("well_id")[fold_columns].reset_index(drop=True)),
        "donor_structure_sha256": _frame_hash(graph.sort_values(["cut_id", "donor_rank"])[donor_columns].reset_index(drop=True)),
        "branch_group_sha256": _frame_hash(groups.sort_values("well_id").reset_index(drop=True)),
        "control_metrics": control_report,
        "next_step": "Replay the frozen V599 strong base on this fixed pseudo-test manifest.",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {"data_dir": str(args.data_dir.resolve()), "artifact_dir": str(args.artifact_dir.resolve()), "run_id": args.run_id, "limit_wells": args.limit_wells}
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
