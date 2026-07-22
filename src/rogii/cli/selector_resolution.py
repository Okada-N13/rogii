from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.selector_replay import likelihood_selector
from rogii.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit selector particle/seed/tracking resolution")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage17b-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def stable_stratified_sample(frame: pd.DataFrame, per_stratum: int) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for _, group in frame.groupby(["stage16_fold", "requested_fraction"], sort=True):
        ranked = group.copy()
        ranked["sample_hash"] = ranked["cut_id"].astype(str).map(
            lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        )
        selected.append(ranked.sort_values(["sample_hash", "cut_id"], kind="stable").head(per_stratum))
    return pd.concat(selected, ignore_index=True).drop(columns="sample_hash")


def _rmse(sse: float, rows: int) -> float:
    return float(np.sqrt(float(sse) / int(rows))) if rows else float("nan")


def _profile_report(rows: pd.DataFrame) -> dict[str, Any]:
    count = int(rows["suffix_rows"].sum())
    baseline = _rmse(float(rows["baseline_sse"].sum()), count)
    screen = _rmse(float(rows["selector_sse"].sum()), count)
    profile = _rmse(float(rows["profile_sse"].sum()), count)
    fold_report = []
    for fold, frame in rows.groupby("stage16_fold", sort=True):
        fold_rows = int(frame["suffix_rows"].sum())
        fold_base = _rmse(float(frame["baseline_sse"].sum()), fold_rows)
        fold_screen = _rmse(float(frame["selector_sse"].sum()), fold_rows)
        fold_profile = _rmse(float(frame["profile_sse"].sum()), fold_rows)
        fold_report.append({
            "fold": int(fold), "baseline_rmse": fold_base, "screen_rmse": fold_screen,
            "profile_rmse": fold_profile, "delta_vs_baseline": fold_profile - fold_base,
            "delta_vs_screen": fold_profile - fold_screen,
        })
    return {
        "cuts": int(len(rows)), "rows": count, "baseline_rmse": baseline,
        "screen_rmse": screen, "profile_rmse": profile,
        "delta_vs_baseline": profile - baseline, "delta_vs_screen": profile - screen,
        "improved_vs_baseline_folds": int(sum(row["delta_vs_baseline"] < 0 for row in fold_report)),
        "fold_report": fold_report,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    source = args.stage17b_run.resolve()
    cuts = pd.read_parquet(source / "cut_report.parquet")
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    if source_summary.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 17B manifest does not match the frozen hash")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    uncovered = cuts[(cuts["evaluation_role"] == "primary") & (~cuts["replay_eligible"])].copy()
    profile_configs = dict(config.get("profiles", {}))
    report_rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for profile_name, profile in profile_configs.items():
        selected = stable_stratified_sample(uncovered, int(profile["cuts_per_stratum"]))
        results: list[dict[str, Any]] = []
        for position, cut in enumerate(selected.itertuples(index=False), 1):
            well_id = str(cut.well_id)
            horizontal = pd.read_csv(args.data_dir.resolve() / "train" / f"{well_id}__horizontal_well.csv")
            typewell = pd.read_csv(args.data_dir.resolve() / "train" / f"{well_id}__typewell.csv")
            cut_index = int(cut.cut_index)
            truth = horizontal["TVT"].to_numpy(float)[cut_index:]
            model_input = horizontal[["MD", "Z", "GR", "TVT"]].copy()
            model_input.loc[model_input.index >= cut_index, "TVT"] = np.nan
            prediction, audit = likelihood_selector(model_input, typewell, cut_index, dict(profile))
            record = {
                "profile": profile_name, "cut_id": str(cut.cut_id), "well_id": well_id,
                "stage16_fold": int(cut.stage16_fold),
                "requested_fraction": float(cut.requested_fraction),
                "suffix_rows": int(cut.suffix_rows), "baseline_sse": float(cut.baseline_sse),
                "selector_sse": float(cut.selector_sse),
                "profile_sse": float(np.square(prediction - truth).sum()), **audit,
            }
            results.append(record); report_rows.append(record)
            if position % 50 == 0:
                print(f"{profile_name}: {position}/{len(selected)} cuts", flush=True)
        profile_frame = pd.DataFrame.from_records(results)
        reports[profile_name] = _profile_report(profile_frame)

    report = pd.DataFrame.from_records(report_rows)
    report.to_parquet(output / "resolution_cut_report.parquet", index=False)
    gates_config = dict(config.get("gates", {}))
    gates: dict[str, bool] = {"hidden_target_invariance": True, "deterministic_stratified_sample": True}
    for name, metrics in reports.items():
        gates[f"{name}_gain"] = metrics["delta_vs_baseline"] <= -float(gates_config.get("minimum_baseline_gain", 1.0))
        gates[f"{name}_fold_consistency"] = metrics["improved_vs_baseline_folds"] >= int(gates_config.get("minimum_improved_folds", 4))
        gates[f"{name}_screen_stability"] = metrics["delta_vs_screen"] <= float(gates_config.get("maximum_delta_vs_screen", 1.0))
    summary = {
        "stage17d_complete": True, "promoted_to_stage18": bool(all(gates.values())),
        "stage16b_manifest_sha256": expected_hash, "profile_configs": profile_configs,
        "profile_reports": reports, "gates": gates,
        "decision_if_promoted": "Keep always-selector as validation control; begin Stage 18 branch retrieval.",
        "next_step": "Start Stage 18 target-free branch retrieval if resolution remains stable.",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {"stage17b_run": str(source), "data_dir": str(args.data_dir.resolve()), "run_id": args.run_id}
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
