from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.branch_retrieval import _bootstrap, _metrics
from rogii.cli.prefix_router import FOLD_FAMILIES, build_candidates
from rogii.cli.trajectory_residual import _typewell_folds
from rogii.config import load_config
from rogii.models.spatial_surface_state import anchored_spatial_plane, guarded_spatial_blend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 27A target-safe XY surface-state audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--public-oof-run", type=Path, required=True)
    parser.add_argument("--stage24a-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _group_improvements(frame: pd.DataFrame, family: str, column: str) -> dict[str, Any]:
    rows = []
    for fold, group in frame.groupby(family, sort=True):
        count = int(group["suffix_rows"].sum())
        base = float(np.sqrt(group["base_sse"].sum() / count))
        candidate = float(np.sqrt(group[column].sum() / count))
        rows.append({"fold": int(fold), "base_rmse": base, "candidate_rmse": candidate, "delta": candidate - base})
    return {"improved_groups": sum(row["delta"] < 0 for row in rows), "groups": len(rows), "report": rows}


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 5 or np.std(left[finite]) < 1e-9 or np.std(right[finite]) < 1e-9:
        return 0.0
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    stage16, stage17 = args.stage16b_run.resolve(), args.stage17a_run.resolve()
    public_run, stage24 = args.public_oof_run.resolve(), args.stage24a_run.resolve()
    validation_run = args.validation_run.resolve()
    summary24 = json.loads((stage24 / "summary.json").read_text(encoding="utf-8"))
    if summary24.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 24A provenance mismatch")
    if not summary24.get("gates", {}).get("hidden_target_invariance"):
        raise AssertionError("Stage 24A hidden-target invariance failed")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_parquet(stage17 / "cut_report.parquet")
    validation_ids = pd.read_parquet(validation_run / "confidence_cut_report.parquet", columns=["cut_id"])
    validation = cuts[cuts["cut_id"].isin(validation_ids["cut_id"])].copy().sort_values("cut_id")
    assignments = pd.read_parquet(stage16 / "well_assignments.parquet")
    assignments["well_id"] = assignments["well_id"].astype(str)
    assignments["typewell_fold"] = _typewell_folds(assignments, 5, seed)
    validation = validation.merge(
        assignments[["well_id", "spatial_fold", "typewell_fold", "branch_group_fold"]],
        on="well_id", how="left", validate="many_to_one",
    )
    validation["stage16_fold"] = validation["stage16_fold"].astype(int)
    wells = sorted(validation["well_id"].astype(str).unique())
    public = pd.read_parquet(
        public_run / "base_oof.parquet",
        columns=["well_id", "row_index", "y_pred"],
        filters=[("well_id", "in", wells)],
    )
    public["well_id"] = public["well_id"].astype(str)
    public_by_well = {well: frame.sort_values("row_index") for well, frame in public.groupby("well_id")}
    train_dir = args.data_dir.resolve() / "train"

    @lru_cache(maxsize=64)
    def load_well(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train_dir / f"{well_id}__horizontal_well.csv")

    @lru_cache(maxsize=64)
    def load_typewell(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train_dir / f"{well_id}__typewell.csv")

    candidate_config = dict(config.get("candidates", {}))
    plane_configs = [dict(value) for value in config.get("plane_profiles", [])]
    blend_profiles = [dict(value) for value in config.get("blend_profiles", [])]
    primary = str(config["primary_profile"])
    rows = []
    invariance = True
    for position, record in enumerate(validation.itertuples(index=False), 1):
        well_id, cut = str(record.well_id), int(record.cut_index)
        horizontal = load_well(well_id)
        source = public_by_well[well_id]
        original = int(source["row_index"].min())
        public_prediction = source["y_pred"].to_numpy(float)[cut - original :]
        base = build_candidates(
            horizontal, load_typewell(well_id), cut, public_prediction, candidate_config
        )[str(config.get("base_candidate", "top_pf_a130"))]
        truth = horizontal["TVT"].to_numpy(float)[cut:]
        suffix = horizontal.iloc[cut:]
        anchor = horizontal.iloc[cut - 1]
        item: dict[str, Any] = {
            "cut_id": str(record.cut_id), "well_id": well_id,
            "requested_fraction": float(record.requested_fraction),
            "stage16_fold": int(record.stage16_fold), "spatial_fold": int(record.spatial_fold),
            "typewell_fold": int(record.typewell_fold),
            "branch_group_fold": int(record.branch_group_fold),
            "suffix_rows": len(truth), "base_sse": float(np.square(base - truth).sum()),
        }
        end_xy = suffix[["X", "Y"]].to_numpy(float)[-1] - np.array([float(anchor["X"]), float(anchor["Y"])])
        true_end_change = float(truth[-1] + suffix["Z"].to_numpy(float)[-1] - anchor["TVT"] - anchor["Z"])
        item["true_end_u_change"] = true_end_change
        plane_predictions: dict[str, np.ndarray] = {}
        for plane_config in plane_configs:
            name = str(plane_config["name"])
            plane, gradient = anchored_spatial_plane(
                horizontal, cut, window_ft=float(plane_config["window_ft"]),
                ridge=float(plane_config.get("ridge", 0.05)),
            )
            plane_predictions[name] = plane
            item[f"plane_{name}_end_u_change"] = float(end_xy @ gradient)
        changed = horizontal.copy()
        changed.loc[changed.index[cut:], "TVT"] += 997.0
        check, _ = anchored_spatial_plane(
            changed, cut, window_ft=float(plane_configs[0]["window_ft"]),
            ridge=float(plane_configs[0].get("ridge", 0.05)),
        )
        invariance = invariance and np.array_equal(check, plane_predictions[str(plane_configs[0]["name"])])
        candidate_sses = []
        for profile in blend_profiles:
            name = str(profile["name"])
            prediction = guarded_spatial_blend(
                base, plane_predictions[str(profile["plane"])],
                weight=float(profile["weight"]), cap_ft=float(profile["cap_ft"]),
                ramp_rows=float(profile.get("ramp_rows", 96)),
            )
            sse = float(np.square(prediction - truth).sum())
            item[f"candidate_sse_{name}"] = sse
            candidate_sses.append(sse)
        item["oracle_sse"] = min([item["base_sse"], *candidate_sses])
        rows.append(item)
        if position % 10 == 0:
            print(f"spatial surface states {position}/{len(validation)} cuts", flush=True)

    frame = pd.DataFrame(rows)
    profile_report = []
    for profile in blend_profiles:
        name = str(profile["name"])
        metric_frame = frame.rename(columns={f"candidate_sse_{name}": "candidate_sse"})
        metrics = _metrics(metric_frame)
        metrics.update({"profile": name, "plane": str(profile["plane"])})
        profile_report.append(metrics)
    primary_column = f"candidate_sse_{primary}"
    primary_frame = frame.rename(columns={primary_column: "candidate_sse"})
    primary_metrics = _metrics(primary_frame)
    bootstrap = _bootstrap(primary_frame, int(config.get("bootstrap_resamples", 4000)), seed)
    count = int(frame["suffix_rows"].sum())
    oracle_rmse = float(np.sqrt(frame["oracle_sse"].sum() / count))
    base_rmse = float(np.sqrt(frame["base_sse"].sum() / count))
    correlations = {
        str(plane["name"]): _safe_correlation(
            frame[f"plane_{plane['name']}_end_u_change"].to_numpy(float),
            frame["true_end_u_change"].to_numpy(float),
        )
        for plane in plane_configs
    }
    group_reports = {
        family: _group_improvements(frame, family, primary_column)
        for family in (*FOLD_FAMILIES, "requested_fraction")
    }
    gate = dict(config.get("gates", {}))
    gates = {
        "hidden_target_invariance": bool(invariance),
        "observable_slope_signal": max(correlations.values()) >= float(gate.get("minimum_end_change_correlation", 0.15)),
        "oracle_headroom": oracle_rmse - base_rmse <= -float(gate.get("minimum_oracle_gain", 0.10)),
        "primary_gain": primary_metrics["rmse_delta"] <= -float(gate.get("minimum_primary_gain", 0.03)),
        "bootstrap_upper_below_zero": bootstrap[1] < 0.0,
        "p90_nonworse": primary_metrics["cut_rmse_p90_delta"] <= 0.0,
        "standard_consistency": group_reports["stage16_fold"]["improved_groups"] >= 4,
        "spatial_consistency": group_reports["spatial_fold"]["improved_groups"] >= 4,
        "typewell_consistency": group_reports["typewell_fold"]["improved_groups"] >= 4,
        "branch_consistency": group_reports["branch_group_fold"]["improved_groups"] >= 4,
        "fraction_consistency": group_reports["requested_fraction"]["improved_groups"] >= 3,
    }
    promoted = bool(all(gates.values()))
    frame.to_parquet(output / "spatial_surface_report.parquet", index=False)
    summary = {
        "stage27a_complete": True,
        "promoted_to_stage27b_spatial_surface_model": promoted,
        "stage16b_manifest_sha256": expected_hash,
        "cuts": len(frame), "wells": int(frame["well_id"].nunique()), "rows": count,
        "base_rmse": base_rmse, "oracle_rmse": oracle_rmse,
        "oracle_delta": oracle_rmse - base_rmse,
        "primary_profile": primary, "primary_metrics": primary_metrics,
        "bootstrap_95pct": bootstrap, "end_u_change_correlations": correlations,
        "group_reports": group_reports, "profile_report": profile_report,
        "gates": gates, "reserved_confirmation_used": False,
        "next_step": (
            "Train a fold-safe spatial-surface state model on the 500 training cuts."
            if promoted else
            "Reject prefix-only XY plane continuation; preserve the 120-well confirmation reserve."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16), "stage17a_run": str(stage17),
        "public_oof_run": str(public_run), "stage24a_run": str(stage24),
        "validation_run": str(validation_run), "data_dir": str(args.data_dir.resolve()),
        "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()

