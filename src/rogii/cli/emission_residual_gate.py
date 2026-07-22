from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.evaluation.delta_u_gate import absolute_tail_metrics, nested_select_predictions, prediction_report
from rogii.evaluation.metrics import evaluate_predictions, paired_well_bootstrap


FAMILIES = ("fold", "spatial_fold", "typewell_fold")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 14B extended residual correction gate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage14-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--limit-wells", type=int)
    return parser


def _load_family(root: Path, family: str, limit_wells: int | None) -> pd.DataFrame:
    frame = pd.read_parquet(root / f"{family}_residual_oof.parquet")
    if limit_wells is not None:
        wells = frame["well_id"].drop_duplicates().iloc[:limit_wells]
        frame = frame[frame["well_id"].isin(set(wells))].copy()
    required = {"id", "well_id", "MD", "y_true", "y_pred", "fold", "raw_generic_residual"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{family} Stage 14 artifact is missing columns: {missing}")
    if family == "fold" and "raw_stacked_residual" not in frame:
        raise ValueError("Standard Stage 14 artifact is missing stacked residual")
    return frame.sort_values("id").reset_index(drop=True)


def _candidate_predictions(frame: pd.DataFrame, specs: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    base = frame["y_pred"].to_numpy(float)
    branches = {"generic": frame["raw_generic_residual"].to_numpy(float)}
    if "raw_stacked_residual" in frame:
        branches["stacked"] = frame["raw_stacked_residual"].to_numpy(float)
    output = {}
    for branch, correction in branches.items():
        for spec in specs:
            output[f"{branch}_{spec['name']}"] = base + float(spec["weight"]) * np.clip(
                correction, -float(spec["cap"]), float(spec["cap"])
            )
    return output


def _family_report(base: pd.DataFrame, predictions: dict[str, np.ndarray], selection: dict[str, Any]):
    nested, selections = nested_select_predictions(base, predictions, selection)
    base_metrics, _ = evaluate_predictions(base)
    nested_metrics, _ = evaluate_predictions(nested)
    profile_reports = {name: prediction_report(base, values) for name, values in predictions.items()}
    bootstrap = paired_well_bootstrap(
        nested, base, n_resamples=int(selection.get("bootstrap_resamples", 3000)), seed=42
    )
    return nested, {
        "base_metrics": base_metrics,
        "nested_metrics": nested_metrics,
        "nested_rmse_delta": float(nested_metrics["pooled_rmse"] - base_metrics["pooled_rmse"]),
        "base_tail": absolute_tail_metrics(base),
        "nested_tail": absolute_tail_metrics(nested),
        "bootstrap": bootstrap,
        "selections": selections,
        "profile_reports": profile_reports,
    }


def _absolute_tail_safe(report: dict[str, Any], tolerance: float) -> bool:
    candidate = report["candidate_tail"]
    base = report["base_tail"]
    return bool(
        candidate["worst_tail_sse"] <= base["worst_tail_sse"] * (1.0 + tolerance)
        and candidate["well_rmse_cvar"] <= base["well_rmse_cvar"] * (1.0 + tolerance)
        and candidate["well_rmse_p90"] <= base["well_rmse_p90"] * (1.0 + tolerance)
        and candidate["well_rmse_max"] <= base["well_rmse_max"] * (1.0 + tolerance)
    )


def _choose_robust_spec(
    reports: dict[str, Any], specs: list[dict[str, Any]], fold_tolerance: float, tail_tolerance: float
) -> str | None:
    eligible: list[tuple[float, str]] = []
    for spec in specs:
        name = f"generic_{spec['name']}"
        family_profiles = [reports[family]["profile_reports"][name] for family in FAMILIES]
        if all(
            profile["pooled_rmse_delta"] < 0.0
            and max(profile["fold_deltas"].values(), default=0.0) <= fold_tolerance
            and _absolute_tail_safe(profile, tail_tolerance)
            for profile in family_profiles
        ):
            eligible.append((sum(float(profile["pooled_rmse_delta"]) for profile in family_profiles), name))
    return min(eligible)[1] if eligible else None


def _tail_delta(candidate: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(candidate[key] - base[key])
        for key in ("worst_tail_sse", "well_rmse_cvar", "well_rmse_p90", "well_rmse_max")
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    specs = [dict(row) for row in config.get("correction_specs", [])]
    selection = dict(config.get("selection", {}))
    validation = dict(config.get("validation", {}))
    if not specs or len({row["name"] for row in specs}) != len(specs):
        raise ValueError("Stage 14B requires unique correction specs")
    stage14 = args.stage14_run.resolve()
    summary14 = load_config(stage14 / "gate_summary.json")
    if summary14.get("promoted_to_full_residual_training"):
        raise RuntimeError("Stage 14 already promoted; Stage 14B audit is unnecessary")
    if summary14.get("robust_generic_spec") != "generic_w050_cap8":
        raise RuntimeError("Stage 14B expects the boundary optimum generic_w050_cap8")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    frames = {family: _load_family(stage14, family, args.limit_wells) for family in FAMILIES}
    ids = frames["fold"]["id"].to_numpy(str)
    alignment = {
        family: {
            "rows": int(len(frame)),
            "id_order_matches": bool(np.array_equal(frame["id"].to_numpy(str), ids)),
            "target_max_abs_difference": float(np.max(np.abs(frame["y_true"].to_numpy(float) - frames["fold"]["y_true"].to_numpy(float)))),
        }
        for family, frame in frames.items()
    }
    if not all(row["id_order_matches"] and row["target_max_abs_difference"] == 0.0 for row in alignment.values()):
        raise AssertionError(f"Stage 14B alignment failed: {alignment}")

    reports, nested_frames = {}, {}
    for family, frame in frames.items():
        predictions = _candidate_predictions(frame, specs)
        nested, report = _family_report(frame, predictions, selection)
        reports[family] = report
        nested_frames[family] = nested
        artifact = frame[["id", "well_id", "MD", "y_true", "y_pred", "fold"]].copy()
        artifact["nested_y_pred"] = nested["y_pred"].to_numpy(float)
        artifact.to_parquet(output / f"{family}_extended_oof.parquet", index=False)

    robust_name = _choose_robust_spec(
        reports,
        specs,
        float(validation.get("inference_fold_tolerance", 0.0)),
        float(validation.get("profile_tail_tolerance", 0.01)),
    )
    standard, spatial, typewell = (reports[name] for name in FAMILIES)
    standard_tail_delta = _tail_delta(standard["nested_tail"], standard["base_tail"])
    gates = {
        "standard_nested_gain": standard["nested_rmse_delta"] <= -float(validation.get("minimum_standard_gain", 0.30)),
        "spatial_nested_gain": spatial["nested_rmse_delta"] <= -float(validation.get("minimum_holdout_gain", 0.25)),
        "typewell_nested_gain": typewell["nested_rmse_delta"] <= -float(validation.get("minimum_holdout_gain", 0.25)),
        "standard_bootstrap": standard["bootstrap"]["ci_97_5"] < 0.0,
        "spatial_bootstrap": spatial["bootstrap"]["ci_97_5"] < 0.0,
        "typewell_bootstrap": typewell["bootstrap"]["ci_97_5"] < 0.0,
        "standard_absolute_tail_nonworse": all(value <= 0.0 for value in standard_tail_delta.values()),
        "robust_extended_generic_spec": robust_name is not None,
    }
    promoted = all(gates.values())
    summary = {
        "promoted_to_full_residual_training": promoted,
        "experiment": "stage14b_extended_residual_gate",
        "alignment": alignment,
        "family_reports": reports,
        "standard_absolute_tail_delta": standard_tail_delta,
        "robust_extended_generic_spec": robust_name,
        "gates": gates,
        "next_step": "Train full-data emission/residual ensemble and build independent test inference." if promoted else "Retain the Stage 14 safe point and inspect the failed absolute-tail/profile constraint.",
    }
    write_json(output / "gate_summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {"stage14_run": str(stage14), "artifact_dir": str(args.artifact_dir.resolve()), "run_id": args.run_id, "limit_wells": args.limit_wells}
    write_yaml(output / "config.yaml", config)
    print({"promoted_to_full_residual_training": promoted, "standard_delta": standard["nested_rmse_delta"], "spatial_delta": spatial["nested_rmse_delta"], "typewell_delta": typewell["nested_rmse_delta"], "bootstrap_95pct": {family: [report["bootstrap"]["ci_2_5"], report["bootstrap"]["ci_97_5"]] for family, report in reports.items()}, "standard_absolute_tail_delta": standard_tail_delta, "robust_extended_generic_spec": robust_name, "gates": gates, "next_step": summary["next_step"]}, flush=True)
    print(f"run artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
