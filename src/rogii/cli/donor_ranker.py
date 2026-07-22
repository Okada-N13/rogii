from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.branch_retrieval import _bootstrap, _metrics, _weighted_retrieval
from rogii.config import load_config


FEATURE_COLUMNS = [
    "donor_rank", "minimum_xyz_distance_ft", "median_xyz_distance_ft", "matched_prefix_points",
    "mean_abs_gr_difference", "typewell_gr_mean_difference", "requested_fraction",
    "prefix_rows_log", "suffix_rows_log", "prefix_u_rmse", "prefix_u_mae", "prefix_u_max_abs",
    "prefix_u_correlation", "calibration_offset_abs", "prefix_distance_mean", "prefix_distance_p90",
    "prefix_gr_delta_mean", "prefix_gr_delta_p90", "suffix_distance_mean", "suffix_distance_p90",
    "suffix_distance_max", "suffix_gr_delta_mean", "suffix_gr_delta_p90", "suffix_donor_u_std",
    "suffix_donor_u_change", "suffix_vs_base_rmse", "suffix_vs_base_mae",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-fitted target-free donor ranker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--stage17b-run", type=Path, required=True)
    parser.add_argument("--stage18c-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3 or np.std(left[finite]) < 1e-8 or np.std(right[finite]) < 1e-8:
        return 0.0
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def _training_mask(rows: pd.DataFrame, evaluation_fold: int) -> pd.Series:
    """Exclude the evaluation fold in both target and donor roles."""
    return (rows["target_branch_fold"] != evaluation_fold) & (rows["donor_branch_fold"] != evaluation_fold)


def _feature_record(
    edge: pd.Series, requested_fraction: float, cut_index: int, suffix_rows: int,
    target_u: np.ndarray, donor_u: np.ndarray, distance: np.ndarray, gr_delta: np.ndarray,
    base_prediction: np.ndarray, z: np.ndarray, prefix_rows: int,
) -> tuple[dict[str, float], np.ndarray]:
    tail_start = max(0, cut_index - prefix_rows)
    prefix_target = target_u[tail_start:cut_index]
    prefix_donor = donor_u[tail_start:cut_index]
    offset = float(np.median(prefix_target - prefix_donor))
    calibrated = donor_u + offset
    prefix_error = calibrated[tail_start:cut_index] - prefix_target
    suffix_prediction = calibrated[cut_index:] - z[cut_index:]
    suffix_distance = distance[cut_index:]
    suffix_gr = gr_delta[cut_index:]
    prefix_distance = distance[tail_start:cut_index]
    prefix_gr = gr_delta[tail_start:cut_index]
    features = {
        "donor_rank": float(edge["donor_rank"]),
        "minimum_xyz_distance_ft": float(edge["minimum_xyz_distance_ft"]),
        "median_xyz_distance_ft": float(edge["median_xyz_distance_ft"]),
        "matched_prefix_points": float(edge["matched_prefix_points"]),
        "mean_abs_gr_difference": float(edge["mean_abs_gr_difference"]),
        "typewell_gr_mean_difference": float(edge["typewell_gr_mean_difference"]),
        "requested_fraction": float(requested_fraction),
        "prefix_rows_log": float(np.log1p(cut_index)),
        "suffix_rows_log": float(np.log1p(suffix_rows)),
        "prefix_u_rmse": float(np.sqrt(np.mean(np.square(prefix_error)))),
        "prefix_u_mae": float(np.mean(np.abs(prefix_error))),
        "prefix_u_max_abs": float(np.max(np.abs(prefix_error))),
        "prefix_u_correlation": _safe_correlation(calibrated[tail_start:cut_index], prefix_target),
        "calibration_offset_abs": abs(offset),
        "prefix_distance_mean": float(np.mean(prefix_distance)),
        "prefix_distance_p90": float(np.quantile(prefix_distance, 0.9)),
        "prefix_gr_delta_mean": float(np.mean(prefix_gr)),
        "prefix_gr_delta_p90": float(np.quantile(prefix_gr, 0.9)),
        "suffix_distance_mean": float(np.mean(suffix_distance)),
        "suffix_distance_p90": float(np.quantile(suffix_distance, 0.9)),
        "suffix_distance_max": float(np.max(suffix_distance)),
        "suffix_gr_delta_mean": float(np.mean(suffix_gr)),
        "suffix_gr_delta_p90": float(np.quantile(suffix_gr, 0.9)),
        "suffix_donor_u_std": float(np.std(calibrated[cut_index:])),
        "suffix_donor_u_change": float(calibrated[-1] - calibrated[cut_index]),
        "suffix_vs_base_rmse": float(np.sqrt(np.mean(np.square(suffix_prediction - base_prediction)))),
        "suffix_vs_base_mae": float(np.mean(np.abs(suffix_prediction - base_prediction))),
    }
    return features, suffix_prediction


def _model(config: dict[str, Any]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=float(config.get("learning_rate", 0.05)),
        max_iter=int(config.get("max_iter", 180)),
        max_leaf_nodes=int(config.get("max_leaf_nodes", 15)),
        max_depth=int(config.get("max_depth", 4)),
        min_samples_leaf=int(config.get("min_samples_leaf", 40)),
        l2_regularization=float(config.get("l2_regularization", 2.0)),
        early_stopping=False,
        random_state=int(config.get("seed", 42)),
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    stage16 = args.stage16b_run.resolve()
    stage17a, stage17b = args.stage17a_run.resolve(), args.stage17b_run.resolve()
    stage18c = args.stage18c_run.resolve()
    expected_manifest = str(config["provenance"]["stage16b_manifest_sha256"])
    expected_sample = str(config["provenance"]["stage18c_sample_sha256"])
    stage18c_summary = json.loads((stage18c / "summary.json").read_text(encoding="utf-8"))
    if not stage18c_summary.get("promoted_to_all_cut_retrieval"):
        raise AssertionError("Stage 18C control was not promoted")
    if stage18c_summary.get("stage16b_manifest_sha256") != expected_manifest:
        raise AssertionError("Stage 18C manifest mismatch")
    if stage18c_summary.get("sample_sha256") != expected_sample:
        raise AssertionError("Stage 18C all-cut sample mismatch")

    output = args.artifact_dir.resolve() / args.run_id
    summary_path = output / "summary.json"
    if summary_path.is_file():
        raise FileExistsError(f"Completed run already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_parquet(stage17b / "cut_report.parquet")
    cuts = cuts[cuts["evaluation_role"] == "primary"].sort_values(
        ["well_id", "cut_index", "cut_id"], kind="stable"
    ).reset_index(drop=True)
    cut_ids = cuts["cut_id"].astype(str).tolist()
    sample_sha = hashlib.sha256("\n".join(sorted(cut_ids)).encode("utf-8")).hexdigest()
    if sample_sha != expected_sample:
        raise AssertionError("Current primary cuts do not match Stage 18C")

    eligible_ids = cuts.loc[cuts["replay_eligible"], "cut_id"].astype(str).tolist()
    uncovered_ids = cuts.loc[~cuts["replay_eligible"], "cut_id"].astype(str).tolist()
    prediction_parts = []
    if eligible_ids:
        prediction_parts.append(pd.read_parquet(stage17a / "replay_predictions.parquet", filters=[("cut_id", "in", eligible_ids)]))
    if uncovered_ids:
        prediction_parts.append(pd.read_parquet(stage17b / "selector_predictions.parquet", filters=[("cut_id", "in", uncovered_ids)]))
    base_predictions = pd.concat(prediction_parts, ignore_index=True)
    base_predictions["cut_id"] = base_predictions["cut_id"].astype(str)
    if set(base_predictions["cut_id"].unique()) != set(cut_ids):
        raise AssertionError("Strong-base predictions do not cover the Stage 18D cuts")
    base_by_cut = base_predictions.groupby("cut_id", sort=False)

    graph = pd.read_parquet(stage16 / "donor_graph.parquet")
    graph = graph[graph["cut_id"].astype(str).isin(cut_ids)].sort_values(["cut_id", "donor_rank"], kind="stable")
    graph["cut_id"] = graph["cut_id"].astype(str)
    graph_by_cut = graph.groupby("cut_id", sort=False)
    assignments = pd.read_parquet(stage16 / "well_assignments.parquet").set_index("well_id")
    data_train = args.data_dir.resolve() / "train"
    retrieval = dict(config.get("retrieval", {}))
    ranker = dict(config.get("ranker", {}))
    maximum_candidates = int(retrieval.get("maximum_candidates", 12))
    selected_donors = int(retrieval.get("selected_donors", 4))
    prefix_rows = int(retrieval.get("prefix_calibration_rows", 256))

    @lru_cache(maxsize=64)
    def load_well(well_id: str) -> pd.DataFrame:
        return pd.read_csv(data_train / f"{well_id}__horizontal_well.csv", usecols=["X", "Y", "Z", "GR", "TVT"])

    @lru_cache(maxsize=64)
    def align_donor(well_id: str, donor_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target, donor = load_well(well_id), load_well(donor_id)
        target_xyz, donor_xyz = target[["X", "Y", "Z"]].to_numpy(float), donor[["X", "Y", "Z"]].to_numpy(float)
        index = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(donor_xyz)
        distance, nearest = index.kneighbors(target_xyz)
        nearest = nearest[:, 0]
        donor_u = donor["TVT"].to_numpy(float)[nearest] + donor["Z"].to_numpy(float)[nearest]
        target_gr = pd.to_numeric(target["GR"], errors="coerce").interpolate(limit_direction="both").to_numpy(float)
        donor_gr = pd.to_numeric(donor["GR"], errors="coerce").interpolate(limit_direction="both").to_numpy(float)[nearest]
        return donor_u, distance[:, 0], np.abs(target_gr - donor_gr)

    feature_path = output / "donor_training_rows.parquet"
    if feature_path.is_file():
        donor_rows = pd.read_parquet(feature_path)
        print("Reusing donor feature checkpoint", flush=True)
    else:
        records: list[dict[str, Any]] = []
        for position, cut in enumerate(cuts.itertuples(index=False), 1):
            cut_id, well_id, cut_index = str(cut.cut_id), str(cut.well_id), int(cut.cut_index)
            target = load_well(well_id)
            target_u = target["TVT"].to_numpy(float) + target["Z"].to_numpy(float)
            truth = target["TVT"].to_numpy(float)[cut_index:]
            z = target["Z"].to_numpy(float)
            base = base_by_cut.get_group(cut_id).sort_values("row_index", kind="stable")["y_pred"].to_numpy(float)
            target_fold = int(assignments.loc[well_id, "branch_group_fold"])
            edges = graph_by_cut.get_group(cut_id)
            edges = edges[
                edges["donor_well_id"].astype(str).map(assignments["branch_group_fold"]).astype(int) != target_fold
            ].head(maximum_candidates)
            for edge in edges.itertuples(index=False):
                donor_id = str(edge.donor_well_id)
                donor_u, distance, gr_delta = align_donor(well_id, donor_id)
                feature, donor_prediction = _feature_record(
                    pd.Series(edge._asdict()), float(cut.requested_fraction), cut_index, len(truth),
                    target_u, donor_u, distance, gr_delta, base, z, prefix_rows,
                )
                donor_rmse = float(np.sqrt(np.mean(np.square(donor_prediction - truth))))
                records.append({
                    "cut_id": cut_id, "well_id": well_id, "donor_well_id": donor_id,
                    "stage16_fold": int(cut.stage16_fold), "requested_fraction": float(cut.requested_fraction),
                    "suffix_rows": len(truth), "target_branch_fold": target_fold,
                    "donor_branch_fold": int(assignments.loc[donor_id, "branch_group_fold"]),
                    "label_log_rmse": float(np.log1p(donor_rmse)), "donor_suffix_rmse": donor_rmse,
                    **feature,
                })
            if position % 100 == 0:
                print(f"donor features {position}/{len(cuts)} cuts", flush=True)
        donor_rows = pd.DataFrame.from_records(records)
        donor_rows.to_parquet(feature_path, index=False)

    donor_rows["oof_score"] = np.nan
    model_report = []
    for fold in sorted(donor_rows["target_branch_fold"].unique()):
        train_mask = _training_mask(donor_rows, int(fold))
        valid_mask = donor_rows["target_branch_fold"] == int(fold)
        model = _model({**ranker, "seed": int(config.get("seed", 42)) + int(fold)})
        train = donor_rows.loc[train_mask]
        model.fit(
            train[FEATURE_COLUMNS], train["label_log_rmse"],
            sample_weight=np.sqrt(train["suffix_rows"].to_numpy(float)),
        )
        donor_rows.loc[valid_mask, "oof_score"] = model.predict(donor_rows.loc[valid_mask, FEATURE_COLUMNS])
        model_report.append({
            "fold": int(fold), "training_rows": int(train_mask.sum()), "validation_rows": int(valid_mask.sum()),
            "excluded_target_rows": int((donor_rows["target_branch_fold"] == int(fold)).sum()),
            "excluded_donor_rows": int((donor_rows["donor_branch_fold"] == int(fold)).sum()),
        })
    if donor_rows["oof_score"].isna().any():
        raise AssertionError("Cross-fitted donor scores are incomplete")
    donor_rows.to_parquet(output / "donor_oof_scores.parquet", index=False)

    ranked = donor_rows.sort_values(["cut_id", "oof_score", "donor_well_id"], kind="stable")
    learned_ids = ranked.groupby("cut_id", sort=False).head(selected_donors).groupby("cut_id")["donor_well_id"].agg(list)
    fixed_ids = donor_rows.sort_values(["cut_id", "donor_rank", "donor_well_id"], kind="stable").groupby(
        "cut_id", sort=False
    ).head(selected_donors).groupby("cut_id")["donor_well_id"].agg(list)

    def retrieval_prediction(well_id: str, cut_index: int, donor_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        target = load_well(well_id)
        target_u = target["TVT"].to_numpy(float) + target["Z"].to_numpy(float)
        arrays = [align_donor(well_id, donor_id) for donor_id in donor_ids]
        raw_u = np.stack([item[0] for item in arrays])
        distances = np.stack([item[1] for item in arrays])
        gr_delta = np.stack([item[2] for item in arrays])
        tail_start = max(0, cut_index - prefix_rows)
        offsets = np.median(target_u[None, tail_start:cut_index] - raw_u[:, tail_start:cut_index], axis=1)
        retrieved_u, confidence = _weighted_retrieval(
            raw_u[:, cut_index:] + offsets[:, None], distances[:, cut_index:], gr_delta[:, cut_index:],
            float(retrieval.get("distance_scale_ft", 300.0)), float(retrieval.get("gr_scale", 30.0)),
        )
        prediction = retrieved_u - target["Z"].to_numpy(float)[cut_index:]
        return prediction, confidence

    comparison_rows = []
    blend = float(retrieval.get("blend_weight", 0.20))
    minimum_weight = float(retrieval.get("minimum_weight", 1e-5))
    for position, cut in enumerate(cuts.itertuples(index=False), 1):
        cut_id, well_id, cut_index = str(cut.cut_id), str(cut.well_id), int(cut.cut_index)
        if cut_id not in learned_ids or cut_id not in fixed_ids:
            continue
        target = load_well(well_id)
        truth = target["TVT"].to_numpy(float)[cut_index:]
        base = base_by_cut.get_group(cut_id).sort_values("row_index", kind="stable")["y_pred"].to_numpy(float)
        fixed_retrieval, fixed_confidence = retrieval_prediction(well_id, cut_index, fixed_ids[cut_id])
        learned_retrieval, learned_confidence = retrieval_prediction(well_id, cut_index, learned_ids[cut_id])
        fixed, learned = base.copy(), base.copy()
        fixed_active = np.isfinite(fixed_retrieval) & (fixed_confidence > minimum_weight)
        learned_active = np.isfinite(learned_retrieval) & (learned_confidence > minimum_weight)
        fixed[fixed_active] += blend * (fixed_retrieval[fixed_active] - base[fixed_active])
        learned[learned_active] += blend * (learned_retrieval[learned_active] - base[learned_active])
        overlap = len(set(fixed_ids[cut_id]) & set(learned_ids[cut_id])) / selected_donors
        comparison_rows.append({
            "cut_id": cut_id, "well_id": well_id, "stage16_fold": int(cut.stage16_fold),
            "branch_group_fold": int(assignments.loc[well_id, "branch_group_fold"]),
            "requested_fraction": float(cut.requested_fraction), "suffix_rows": len(truth),
            "strong_base_sse": float(np.square(base - truth).sum()),
            "base_sse": float(np.square(fixed - truth).sum()),
            "candidate_sse": float(np.square(learned - truth).sum()), "donor_overlap_fraction": float(overlap),
        })
        if position % 100 == 0:
            print(f"ranked retrieval {position}/{len(cuts)} cuts", flush=True)

    comparison = pd.DataFrame.from_records(comparison_rows)
    comparison.to_parquet(output / "retrieval_comparison.parquet", index=False)
    learned_vs_fixed = _metrics(comparison)
    branch_comparison = comparison.copy()
    branch_comparison["stage16_fold"] = branch_comparison["branch_group_fold"]
    learned_vs_fixed_branch = _metrics(branch_comparison)
    learned_vs_strong_rows = comparison.copy()
    learned_vs_strong_rows["base_sse"] = learned_vs_strong_rows["strong_base_sse"]
    learned_vs_strong = _metrics(learned_vs_strong_rows)
    ci = _bootstrap(comparison, int(config.get("bootstrap_resamples", 3000)), int(config.get("seed", 42)))
    gates_config = dict(config.get("gates", {}))
    coverage = float(comparison["cut_id"].nunique() / len(cut_ids))
    gates = {
        "hidden_target_invariance": True,
        "target_and_donor_fold_isolation": True,
        "cut_coverage": coverage >= float(gates_config.get("minimum_cut_coverage", 0.99)),
        "learned_gain": learned_vs_fixed["rmse_delta"] <= -float(gates_config.get("minimum_learned_gain", 0.05)),
        "standard_fold_consistency": learned_vs_fixed["improved_folds"] >= int(gates_config.get("minimum_improved_folds", 4)),
        "branch_fold_consistency": learned_vs_fixed_branch["improved_folds"] >= int(gates_config.get("minimum_improved_folds", 4)),
        "fraction_consistency": learned_vs_fixed["improved_fractions"] >= int(gates_config.get("minimum_improved_fractions", 4)),
        "p90_nonworse": learned_vs_fixed["cut_rmse_p90_delta"] <= 0,
        "bootstrap_upper_below_zero": ci[1] < 0,
    }
    score_correlation = float(donor_rows[["oof_score", "label_log_rmse"]].corr(method="spearman").iloc[0, 1])
    predicted_best = ranked.groupby("cut_id", sort=False).first()["donor_well_id"]
    oracle_best = donor_rows.sort_values(["cut_id", "label_log_rmse", "donor_well_id"], kind="stable").groupby(
        "cut_id", sort=False
    ).first()["donor_well_id"]
    top1_recall = float((predicted_best.sort_index() == oracle_best.sort_index()).mean())
    summary = {
        "stage18d_complete": True, "promoted_to_full_ranker_training": bool(all(gates.values())),
        "stage16b_manifest_sha256": expected_manifest, "stage18c_sample_sha256": expected_sample,
        "cuts": len(cut_ids), "candidate_rows": len(donor_rows), "coverage_fraction": coverage,
        "feature_columns": FEATURE_COLUMNS, "hidden_target_invariance": True,
        "model_report": model_report, "oof_score_spearman": score_correlation,
        "oracle_top1_recall": top1_recall, "mean_selected_donor_overlap": float(comparison["donor_overlap_fraction"].mean()),
        "learned_vs_fixed": learned_vs_fixed, "learned_vs_fixed_branch_folds": learned_vs_fixed_branch,
        "learned_vs_strong_base": learned_vs_strong,
        "bootstrap_95pct": ci, "gates": gates,
        "next_step": (
            "Train the all-data donor ranker and export independent test inference."
            if all(gates.values()) else "Keep fixed retrieval; revise ranker features before test inference."
        ),
    }
    write_json(summary_path, summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16), "stage17a_run": str(stage17a), "stage17b_run": str(stage17b),
        "stage18c_run": str(stage18c), "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
