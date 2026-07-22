from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fold-safe target-free branch retrieval benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--stage17b-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def stable_sample(frame: pd.DataFrame, per_stratum: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby(["stage16_fold", "requested_fraction"], sort=True):
        ranked = group.copy()
        ranked["sample_hash"] = ranked["cut_id"].astype(str).map(
            lambda value: hashlib.sha256(("stage18:" + value).encode("utf-8")).hexdigest()
        )
        parts.append(ranked.sort_values(["sample_hash", "cut_id"], kind="stable").head(per_stratum))
    return pd.concat(parts, ignore_index=True).drop(columns="sample_hash")


def _rmse(sse: float, rows: int) -> float:
    return float(np.sqrt(float(sse) / int(rows))) if rows else float("nan")


def _weighted_retrieval(
    donor_u: np.ndarray, distances: np.ndarray, gr_delta: np.ndarray,
    distance_scale: float, gr_scale: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.exp(-np.minimum(distances, 5000.0) / distance_scale)
    if gr_scale is not None:
        weights *= np.exp(-np.minimum(gr_delta, 500.0) / gr_scale)
    total = weights.sum(axis=0)
    prediction = np.divide(
        (weights * donor_u).sum(axis=0), total,
        out=np.full(donor_u.shape[1], np.nan, dtype=float), where=total > 1e-12,
    )
    return prediction, total


def _metrics(rows: pd.DataFrame) -> dict[str, Any]:
    count = int(rows["suffix_rows"].sum())
    base = _rmse(float(rows["base_sse"].sum()), count)
    candidate = _rmse(float(rows["candidate_sse"].sum()), count)
    fold_report = []
    for fold, frame in rows.groupby("stage16_fold", sort=True):
        fold_rows = int(frame["suffix_rows"].sum())
        fold_base = _rmse(float(frame["base_sse"].sum()), fold_rows)
        fold_candidate = _rmse(float(frame["candidate_sse"].sum()), fold_rows)
        fold_report.append({"fold": int(fold), "base_rmse": fold_base, "candidate_rmse": fold_candidate, "delta": fold_candidate - fold_base})
    base_cut = np.sqrt(rows["base_sse"] / rows["suffix_rows"])
    candidate_cut = np.sqrt(rows["candidate_sse"] / rows["suffix_rows"])
    return {
        "cuts": len(rows), "rows": count, "base_rmse": base, "candidate_rmse": candidate,
        "rmse_delta": candidate - base,
        "cut_rmse_p90_delta": float(candidate_cut.quantile(0.9) - base_cut.quantile(0.9)),
        "cut_rmse_max_delta": float(candidate_cut.max() - base_cut.max()),
        "improved_folds": int(sum(item["delta"] < 0 for item in fold_report)),
        "fold_report": fold_report,
    }


def _bootstrap(rows: pd.DataFrame, resamples: int, seed: int) -> list[float]:
    grouped = rows.groupby("well_id", sort=True).agg(
        base_sse=("base_sse", "sum"), candidate_sse=("candidate_sse", "sum"), rows=("suffix_rows", "sum")
    )
    delta = np.sqrt(grouped["candidate_sse"] / grouped["rows"]) - np.sqrt(grouped["base_sse"] / grouped["rows"])
    rng = np.random.default_rng(seed)
    samples = rng.choice(delta.to_numpy(float), size=(resamples, len(delta)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    stage16, stage17a, stage17b = args.stage16b_run.resolve(), args.stage17a_run.resolve(), args.stage17b_run.resolve()
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    stage17b_summary = json.loads((stage17b / "summary.json").read_text(encoding="utf-8"))
    if stage17b_summary.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 17B does not use the frozen manifest")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_parquet(stage17b / "cut_report.parquet")
    selected = stable_sample(cuts[cuts["evaluation_role"] == "primary"], int(config.get("cuts_per_stratum", 6)))
    cut_ids = selected["cut_id"].astype(str).tolist()
    eligible_ids = selected.loc[selected["replay_eligible"], "cut_id"].astype(str).tolist()
    uncovered_ids = selected.loc[~selected["replay_eligible"], "cut_id"].astype(str).tolist()
    frames = []
    if eligible_ids:
        frames.append(pd.read_parquet(stage17a / "replay_predictions.parquet", filters=[("cut_id", "in", eligible_ids)]))
    if uncovered_ids:
        frames.append(pd.read_parquet(stage17b / "selector_predictions.parquet", filters=[("cut_id", "in", uncovered_ids)]))
    base_predictions = pd.concat(frames, ignore_index=True)
    if set(base_predictions["cut_id"].astype(str).unique()) != set(cut_ids):
        raise AssertionError("Strong-base predictions do not cover the fixed retrieval sample")

    graph = pd.read_parquet(stage16 / "donor_graph.parquet")
    graph = graph[graph["cut_id"].astype(str).isin(cut_ids)].sort_values(["cut_id", "donor_rank"], kind="stable")
    assignments = pd.read_parquet(stage16 / "well_assignments.parquet").set_index("well_id")
    manifest = pd.read_parquet(stage16 / "pseudo_test_manifest.parquet").set_index("cut_id")
    retrieval_config = dict(config.get("retrieval", {}))
    profiles = dict(config.get("profiles", {}))
    data_train = args.data_dir.resolve() / "train"

    @lru_cache(maxsize=64)
    def load_well(well_id: str) -> pd.DataFrame:
        return pd.read_csv(data_train / f"{well_id}__horizontal_well.csv", usecols=["X", "Y", "Z", "GR", "TVT"])

    result_rows: list[dict[str, Any]] = []
    families = {"standard": "fold", "spatial": "spatial_fold", "branch": "branch_group_fold"}
    for position, cut_id in enumerate(cut_ids, 1):
        cut = selected[selected["cut_id"].astype(str) == cut_id].iloc[0]
        well_id, cut_index = str(cut["well_id"]), int(cut["cut_index"])
        target = load_well(well_id)
        base = base_predictions[base_predictions["cut_id"].astype(str) == cut_id].sort_values("row_index", kind="stable")
        row_index = base["row_index"].to_numpy(np.int64)
        if not np.array_equal(row_index, np.arange(cut_index, len(target))):
            raise AssertionError(f"{cut_id}: base row alignment mismatch")
        truth = target["TVT"].to_numpy(float)[cut_index:]
        if float(np.max(np.abs(truth - base["y_true"].to_numpy(float)))) > 0.005:
            raise AssertionError(f"{cut_id}: base target mismatch")
        base_pred = base["y_pred"].to_numpy(float)
        target_xyz = target[["X", "Y", "Z"]].to_numpy(float)
        target_gr = pd.to_numeric(target["GR"], errors="coerce").interpolate(limit_direction="both").to_numpy(float)
        target_u = target["TVT"].to_numpy(float) + target["Z"].to_numpy(float)
        edge_frame = graph[graph["cut_id"].astype(str) == cut_id]
        mapping: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for donor_id in edge_frame["donor_well_id"].astype(str).unique():
            donor = load_well(donor_id)
            donor_xyz = donor[["X", "Y", "Z"]].to_numpy(float)
            index = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(donor_xyz)
            distance, nearest = index.kneighbors(target_xyz)
            nearest = nearest[:, 0]; distance = distance[:, 0]
            donor_u = donor["TVT"].to_numpy(float)[nearest] + donor["Z"].to_numpy(float)[nearest]
            donor_gr = pd.to_numeric(donor["GR"], errors="coerce").interpolate(limit_direction="both").to_numpy(float)[nearest]
            mapping[donor_id] = (donor_u, distance, np.abs(target_gr - donor_gr))

        for family, fold_column in families.items():
            target_fold = int(assignments.loc[well_id, fold_column])
            admissible = edge_frame[
                edge_frame["donor_well_id"].astype(str).map(assignments[fold_column]).astype(int) != target_fold
            ].head(int(retrieval_config.get("maximum_donors", 4)))
            if len(admissible) < int(retrieval_config.get("minimum_donors", 2)):
                continue
            donor_ids = admissible["donor_well_id"].astype(str).tolist()
            raw_u = np.stack([mapping[donor][0] for donor in donor_ids])
            distances = np.stack([mapping[donor][1] for donor in donor_ids])
            gr_delta = np.stack([mapping[donor][2] for donor in donor_ids])
            tail_start = max(0, cut_index - int(retrieval_config.get("prefix_calibration_rows", 256)))
            offsets = np.median(target_u[None, tail_start:cut_index] - raw_u[:, tail_start:cut_index], axis=1)
            for profile_name, profile in profiles.items():
                donor_u = raw_u + (offsets[:, None] if bool(profile.get("prefix_calibration", False)) else 0.0)
                retrieved_u, confidence = _weighted_retrieval(
                    donor_u[:, cut_index:], distances[:, cut_index:], gr_delta[:, cut_index:],
                    float(retrieval_config.get("distance_scale_ft", 300.0)),
                    float(retrieval_config.get("gr_scale", 30.0)) if bool(profile.get("use_gr", False)) else None,
                )
                retrieval_prediction = retrieved_u - target["Z"].to_numpy(float)[cut_index:]
                finite = np.isfinite(retrieval_prediction) & (confidence > float(retrieval_config.get("minimum_weight", 1e-5)))
                blend = float(profile.get("blend_weight", 0.2))
                candidate = base_pred.copy()
                candidate[finite] += blend * (retrieval_prediction[finite] - base_pred[finite])
                result_rows.append({
                    "family": family, "profile": profile_name, "cut_id": cut_id, "well_id": well_id,
                    "stage16_fold": int(cut["stage16_fold"]), "suffix_rows": len(truth),
                    "donors": len(donor_ids), "active_fraction": float(finite.mean()),
                    "mean_minimum_distance": float(distances[:, cut_index:].min(axis=0).mean()),
                    "base_sse": float(np.square(base_pred - truth).sum()),
                    "candidate_sse": float(np.square(candidate - truth).sum()),
                })
        if position % 25 == 0:
            print(f"retrieval benchmark {position}/{len(cut_ids)} cuts", flush=True)

    results = pd.DataFrame.from_records(result_rows)
    results.to_parquet(output / "retrieval_cut_report.parquet", index=False)
    reports: dict[str, Any] = {}
    for (family, profile), frame in results.groupby(["family", "profile"], sort=True):
        reports[f"{family}__{profile}"] = _metrics(frame)
    primary_profile = str(config.get("primary_profile", "prefix_gr_w020"))
    standard_key, spatial_key, branch_key = (
        f"standard__{primary_profile}", f"spatial__{primary_profile}", f"branch__{primary_profile}"
    )
    primary_rows = results[(results["family"] == "standard") & (results["profile"] == primary_profile)]
    ci = _bootstrap(primary_rows, int(config.get("bootstrap_resamples", 1000)), int(config.get("seed", 42)))
    gates_config = dict(config.get("gates", {}))
    gates = {
        "hidden_target_invariance": True, "same_fold_donor_excluded": True,
        "standard_gain": reports[standard_key]["rmse_delta"] <= -float(gates_config.get("minimum_standard_gain", 0.30)),
        "standard_fold_consistency": reports[standard_key]["improved_folds"] >= int(gates_config.get("minimum_improved_folds", 4)),
        "spatial_gain": reports[spatial_key]["rmse_delta"] < 0,
        "branch_group_gain": reports[branch_key]["rmse_delta"] < 0,
        "p90_nonworse": reports[standard_key]["cut_rmse_p90_delta"] <= 0,
        "bootstrap_upper_below_zero": ci[1] < 0,
    }
    summary = {
        "stage18a_complete": True, "promoted_to_all_cut_retrieval": bool(all(gates.values())),
        "stage16b_manifest_sha256": expected_hash, "sample_cuts": len(cut_ids),
        "primary_profile": primary_profile, "reports": reports,
        "bootstrap_95pct": ci, "gates": gates,
        "next_step": "Run all-cut retrieval and learned donor ranking if the fixed subset promotes.",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {"stage16b_run": str(stage16), "stage17a_run": str(stage17a), "stage17b_run": str(stage17b), "run_id": args.run_id}
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
