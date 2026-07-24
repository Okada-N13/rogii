from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.branch_retrieval import _bootstrap, _metrics
from rogii.cli.prefix_router import FOLD_FAMILIES, _family_report
from rogii.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 34A OOF-selected conservative hard gate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage33a-run", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def apply_hard_gate(frame: pd.DataFrame, threshold: float) -> pd.Series:
    active = frame["predicted_gate"].to_numpy(float) >= float(threshold)
    candidate = np.where(
        active,
        frame["full_sse"].to_numpy(float),
        frame["base_sse"].to_numpy(float),
    )
    return pd.Series(candidate, index=frame.index, dtype=float)


def _training_audit(
    source: pd.DataFrame,
    threshold: float,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    frame = source.copy()
    frame["predicted_gate"] = frame["crossfit_gate"]
    frame["candidate_sse"] = apply_hard_gate(frame, threshold)
    metrics = _metrics(frame)
    bootstrap = _bootstrap(frame, int(config.get("bootstrap_resamples", 4000)), seed)
    active = frame["predicted_gate"].to_numpy(float) >= float(threshold)
    gates = {
        "minimum_gain": metrics["rmse_delta"] <= -float(config.get("minimum_oof_gain", 0.05)),
        "bootstrap_upper_below_zero": bootstrap[1] < 0.0,
        "p90_nonworse": metrics["cut_rmse_p90_delta"] <= 0.0,
        "standard_consistency": metrics["improved_folds"] >= 4,
        "fraction_consistency": metrics["improved_fractions"] >= 3,
        "minimum_active_fraction": float(active.mean()) >= float(
            config.get("minimum_active_fraction", 0.10)
        ),
    }
    return {
        "threshold": float(threshold),
        "active_fraction": float(active.mean()),
        "active_cuts": int(active.sum()),
        "base_rmse": metrics["base_rmse"],
        "candidate_rmse": metrics["candidate_rmse"],
        "rmse_delta": metrics["rmse_delta"],
        "bootstrap_95pct": bootstrap,
        "cut_rmse_p90_delta": metrics["cut_rmse_p90_delta"],
        "improved_folds": metrics["improved_folds"],
        "improved_fractions": metrics["improved_fractions"],
        "gates": gates,
        "eligible": bool(all(gates.values())),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    stage33 = args.stage33a_run.resolve()
    summary33 = json.loads((stage33 / "summary.json").read_text(encoding="utf-8"))
    if not summary33.get("stage33a_complete"):
        raise AssertionError("Stage 33A did not complete")
    if summary33.get("reserved_confirmation_used") is not False:
        raise AssertionError("Stage 33A opened the confirmation reserve")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    training = pd.read_parquet(stage33 / "training_gate_report.parquet")
    validation = pd.read_parquet(stage33 / "validation_gate_report.parquet")
    required_training = {
        "well_id",
        "suffix_rows",
        "base_sse",
        "full_sse",
        "crossfit_gate",
        "stage16_fold",
        "requested_fraction",
    }
    required_validation = required_training - {"crossfit_gate"} | {
        "predicted_gate",
        "spatial_fold",
        "typewell_fold",
        "branch_group_fold",
    }
    if missing := required_training - set(training):
        raise KeyError(f"Stage 33 training report missing columns: {sorted(missing)}")
    if missing := required_validation - set(validation):
        raise KeyError(f"Stage 33 validation report missing columns: {sorted(missing)}")

    selection_config = dict(config.get("selection", {}))
    quantiles = [float(value) for value in selection_config.get("score_quantiles", [])]
    thresholds = sorted(
        {
            float(np.quantile(training["crossfit_gate"].to_numpy(float), quantile))
            for quantile in quantiles
        }
    )
    audits = [
        _training_audit(training, threshold, selection_config, seed)
        for threshold in thresholds
    ]
    eligible = [audit for audit in audits if audit["eligible"]]
    selected = (
        min(eligible, key=lambda audit: (audit["candidate_rmse"], -audit["threshold"]))
        if eligible
        else None
    )
    selected_threshold = float(selected["threshold"]) if selected else None

    report = validation.copy()
    if selected_threshold is None:
        report["candidate_sse"] = report["base_sse"].astype(float)
        report["hard_gate_active"] = False
    else:
        report["candidate_sse"] = apply_hard_gate(report, selected_threshold)
        report["hard_gate_active"] = (
            report["predicted_gate"].to_numpy(float) >= selected_threshold
        )
    metrics = _metrics(report)
    bootstrap = _bootstrap(
        report, int(config.get("validation", {}).get("bootstrap_resamples", 4000)), seed
    )
    family_reports = {family: _family_report(report, family) for family in FOLD_FAMILIES}
    validation_config = dict(config.get("validation", {}))
    gates = {
        "oof_threshold_available": selected_threshold is not None,
        "hidden_target_invariance": bool(summary33["gates"]["hidden_target_invariance"]),
        "training_validation_well_overlap_zero": bool(
            summary33["gates"]["training_validation_well_overlap_zero"]
        ),
        "primary_gain": metrics["rmse_delta"] <= -float(validation_config.get("minimum_gain", 0.10)),
        "bootstrap_upper_below_zero": bootstrap[1] < 0.0,
        "standard_fold_consistency": metrics["improved_folds"] >= 4,
        "fraction_consistency": metrics["improved_fractions"] >= 3,
        "spatial_fold_consistency": family_reports["spatial_fold"]["improved_folds"] >= 4,
        "typewell_fold_consistency": family_reports["typewell_fold"]["improved_folds"] >= 4,
        "branch_fold_consistency": family_reports["branch_group_fold"]["improved_folds"] >= 4,
        "p90_nonworse": metrics["cut_rmse_p90_delta"] <= 0.0,
    }
    promoted = bool(all(gates.values()))
    report.to_parquet(output / "selective_gate_report.parquet", index=False)
    summary = {
        "stage34a_complete": True,
        "promoted_to_stage34b_reserved_confirmation": promoted,
        "stage16b_manifest_sha256": summary33["stage16b_manifest_sha256"],
        "source_stage33_run": str(stage33),
        "selection_role": "training_oof_only",
        "threshold_candidates": len(audits),
        "eligible_thresholds": len(eligible),
        "selected_threshold": selected_threshold,
        "selected_training_audit": selected,
        "threshold_audits": audits,
        "validation_active_fraction": float(report["hard_gate_active"].mean()),
        "validation_active_cuts": int(report["hard_gate_active"].sum()),
        "base_rmse": metrics["base_rmse"],
        "candidate_rmse": metrics["candidate_rmse"],
        "rmse_delta": metrics["rmse_delta"],
        "bootstrap_95pct": bootstrap,
        "cut_rmse_p90_delta": metrics["cut_rmse_p90_delta"],
        "family_reports": family_reports,
        "gates": gates,
        "hidden_target_features_recomputed": False,
        "reserved_confirmation_used": False,
        "next_step": (
            "Freeze the hard gate and run exactly one Stage 34B reserved confirmation."
            if promoted
            else "End the TCN residual family without opening the confirmation reserve."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {"stage33a_run": str(stage33), "run_id": args.run_id}
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
