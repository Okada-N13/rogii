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
    parser = argparse.ArgumentParser(description="Stage 32A lock a predeclared TCN uncertainty profile")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage31a-run", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _audit_profile(
    source: pd.DataFrame,
    name: str,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    candidate_column = f"candidate_sse_{name}"
    if candidate_column not in source:
        raise KeyError(f"Stage 31 report is missing {candidate_column}")
    frame = source.copy()
    frame["candidate_sse"] = frame[candidate_column]
    metrics = _metrics(frame)
    bootstrap = _bootstrap(frame, int(config.get("bootstrap_resamples", 4000)), seed)
    family_reports = {family: _family_report(frame, family) for family in FOLD_FAMILIES}
    gates = {
        "minimum_gain": metrics["rmse_delta"] <= -float(config.get("minimum_gain", 0.10)),
        "bootstrap_upper_below_zero": bootstrap[1] < 0.0,
        "p90_nonworse": metrics["cut_rmse_p90_delta"] <= 0.0,
        "standard_consistency": metrics["improved_folds"] >= 4,
        "fraction_consistency": metrics["improved_fractions"] >= 3,
        "spatial_consistency": family_reports["spatial_fold"]["improved_folds"] >= 4,
        "typewell_consistency": family_reports["typewell_fold"]["improved_folds"] >= 4,
        "branch_consistency": family_reports["branch_group_fold"]["improved_folds"] >= 4,
    }
    return {
        "profile": name,
        "base_rmse": metrics["base_rmse"],
        "candidate_rmse": metrics["candidate_rmse"],
        "rmse_delta": metrics["rmse_delta"],
        "bootstrap_95pct": bootstrap,
        "cut_rmse_p90_delta": metrics["cut_rmse_p90_delta"],
        "improved_folds": metrics["improved_folds"],
        "improved_fractions": metrics["improved_fractions"],
        "family_improved_folds": {
            family: report["improved_folds"] for family, report in family_reports.items()
        },
        "gates": gates,
        "eligible": bool(all(gates.values())),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    stage31 = args.stage31a_run.resolve()
    summary31 = json.loads((stage31 / "summary.json").read_text(encoding="utf-8"))
    if not summary31.get("stage31a_complete"):
        raise AssertionError("Stage 31A did not complete")
    if summary31.get("reserved_confirmation_used") is not False:
        raise AssertionError("Stage 31A opened the confirmation reserve")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = pd.read_parquet(stage31 / "uncertainty_cut_report.parquet")
    profiles = [str(value) for value in config.get("profiles", [])]
    audits = [_audit_profile(report, profile, dict(config.get("gates", {})), seed) for profile in profiles]
    eligible = [item for item in audits if item["eligible"]]
    selected = min(eligible, key=lambda item: (item["candidate_rmse"], item["profile"])) if eligible else None
    selection = selected["profile"] if selected else None
    summary = {
        "stage32a_complete": True,
        "promoted_to_stage32b_reserved_confirmation": selection is not None,
        "stage16b_manifest_sha256": summary31["stage16b_manifest_sha256"],
        "source_stage31_run": str(stage31),
        "profiles_audited": len(audits),
        "eligible_profiles": len(eligible),
        "selected_profile": selection,
        "selected_metrics": selected,
        "profile_audits": audits,
        "hidden_target_features_recomputed": False,
        "reserved_confirmation_used": False,
        "next_step": (
            f"Lock {selection} and run exactly one Stage 32B reserved confirmation audit."
            if selection else
            "Reject all predeclared TCN uncertainty profiles without opening the reserve."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {"stage31a_run": str(stage31), "run_id": args.run_id}
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()

