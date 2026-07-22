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
from rogii.models.emission_uncertainty import apply_uncertainty_profile, prediction_error_correlation, uncertainty_features


FAMILIES = ("fold", "spatial_fold", "typewell_fold")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 13 emission uncertainty gate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage12b-run", type=Path, required=True)
    parser.add_argument("--stage12c-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--limit-wells", type=int)
    return parser


def _load_family(stage12c: Path, family: str, limit_wells: int | None) -> pd.DataFrame:
    frame = pd.read_parquet(stage12c / f"{family}_path_oof.parquet")
    if limit_wells is not None:
        selected = frame["well_id"].drop_duplicates().iloc[:limit_wells]
        frame = frame[frame["well_id"].isin(set(selected))].copy()
    required = {"id", "well_id", "cut_id", "row_index", "MD", "y_true", "surface_y_pred", "y_pred", "fold"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{family} artifact is missing columns: {missing}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{family} artifact contains duplicate IDs")
    return frame.sort_values("id").reset_index(drop=True)


def _report_family(base: pd.DataFrame, predictions: dict[str, np.ndarray], selection: dict[str, Any]):
    nested, selections = nested_select_predictions(base, predictions, selection)
    base_metrics, _ = evaluate_predictions(base)
    nested_metrics, _ = evaluate_predictions(nested)
    reports = {name: prediction_report(base, values) for name, values in predictions.items()}
    bootstrap = paired_well_bootstrap(
        nested, base, n_resamples=int(selection.get("bootstrap_resamples", 2000)), seed=42
    )
    return nested, {
        "base_metrics": base_metrics,
        "nested_metrics": nested_metrics,
        "nested_rmse_delta": float(nested_metrics["pooled_rmse"] - base_metrics["pooled_rmse"]),
        "base_tail": absolute_tail_metrics(base),
        "nested_tail": absolute_tail_metrics(nested),
        "bootstrap": bootstrap,
        "selections": selections,
        "profile_reports": reports,
    }


def _choose_robust_local(family_reports: dict[str, Any], profiles: list[dict[str, Any]], tolerance: float) -> str | None:
    eligible: list[tuple[float, str]] = []
    for profile in profiles:
        name = str(profile["name"])
        reports = [family_reports[family]["profile_reports"][name] for family in FAMILIES]
        if all(
            report["pooled_rmse_delta"] < 0.0
            and max(report["fold_deltas"].values(), default=0.0) <= tolerance
            for report in reports
        ):
            eligible.append((sum(float(report["pooled_rmse_delta"]) for report in reports), name))
    return min(eligible)[1] if eligible else None


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    profiles = [dict(value) for value in config.get("local_profiles", [])]
    ensembles = [dict(value) for value in config.get("standard_ensemble_profiles", [])]
    selection = dict(config.get("selection", {}))
    validation = dict(config.get("validation", {}))
    if not profiles or len({row["name"] for row in profiles}) != len(profiles):
        raise ValueError("Stage 13 requires unique local profiles")
    stage12b, stage12c = args.stage12b_run.resolve(), args.stage12c_run.resolve()
    summary12b = load_config(stage12b / "gate_summary.json")
    summary12c = load_config(stage12c / "gate_summary.json")
    if not summary12b.get("promoted_to_spatial_emission_audit"):
        raise RuntimeError("Stage 12B did not authorize uncertainty analysis")
    if summary12c.get("promoted_to_all_train_alignment"):
        raise RuntimeError("Stage 12C already promoted; Stage 13 fallback should not replace it")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    frames = {family: _load_family(stage12c, family, args.limit_wells) for family in FAMILIES}
    reference_ids = frames["fold"]["id"].to_numpy(str)
    alignment = {}
    for family, frame in frames.items():
        alignment[family] = {
            "rows": int(len(frame)),
            "id_order_matches": bool(np.array_equal(frame["id"].to_numpy(str), reference_ids)),
            "target_max_abs_difference": float(np.max(np.abs(frame["y_true"].to_numpy(float) - frames["fold"]["y_true"].to_numpy(float)))),
        }
    if not all(row["id_order_matches"] and row["target_max_abs_difference"] == 0.0 for row in alignment.values()):
        raise AssertionError(f"Stage 13 family alignment failed: {alignment}")

    entropy = pd.read_parquet(stage12b / "oof_emission.parquet", columns=["id", "learned_entropy"])
    if entropy["id"].duplicated().any():
        raise ValueError("Stage 12B entropy artifact contains duplicate IDs")
    entropy = entropy.set_index("id").reindex(reference_ids)
    if entropy["learned_entropy"].isna().any():
        raise AssertionError("Stage 12B entropy does not align with Stage 12C rows")

    family_reports, nested_frames = {}, {}
    local_predictions: dict[str, dict[str, np.ndarray]] = {}
    for family, frame in frames.items():
        features = uncertainty_features(frame)
        predictions = {
            str(profile["name"]): apply_uncertainty_profile(frame, features, profile)
            for profile in profiles
        }
        local_predictions[family] = predictions
        if family == "fold":
            standard = frame["y_pred"].to_numpy(float)
            spatial = frames["spatial_fold"]["y_pred"].to_numpy(float)
            typewell = frames["typewell_fold"]["y_pred"].to_numpy(float)
            surface = frame["surface_y_pred"].to_numpy(float)
            disagreement = np.std(np.column_stack([standard, spatial, typewell]), axis=1)
            disagreement_pct = pd.Series(disagreement).rank(pct=True).to_numpy(float)
            entropy_pct = entropy["learned_entropy"].rank(pct=True).to_numpy(float)
            for profile in ensembles:
                candidate = (
                    float(profile.get("standard_weight", 1.0)) * standard
                    + float(profile.get("spatial_weight", 0.0)) * spatial
                    + float(profile.get("typewell_weight", 0.0)) * typewell
                )
                risk = 0.5 * disagreement_pct + 0.5 * entropy_pct
                mask = risk >= float(profile.get("risk_threshold", 1.1))
                candidate[mask] = surface[mask] + float(profile.get("high_risk_scale", 1.0)) * (candidate[mask] - surface[mask])
                predictions[str(profile["name"])] = candidate
        nested, report = _report_family(frame, predictions, selection)
        family_reports[family] = report
        nested_frames[family] = nested
        artifact = frame[["id", "well_id", "cut_id", "row_index", "MD", "y_true", "surface_y_pred", "y_pred", "fold"]].copy()
        artifact["nested_y_pred"] = nested["y_pred"].to_numpy(float)
        for name, values in predictions.items():
            artifact[f"{name}_y_pred"] = values
        artifact.to_parquet(output / f"{family}_gate_oof.parquet", index=False)

    robust_name = _choose_robust_local(
        family_reports, profiles, float(validation.get("inference_fold_tolerance", 0.03))
    )
    standard = family_reports["fold"]
    spatial = family_reports["spatial_fold"]
    typewell = family_reports["typewell_fold"]
    correlation_frame = frames["fold"][["y_true"]].copy()
    correlation_frame["standard"] = frames["fold"]["y_pred"].to_numpy(float)
    correlation_frame["spatial"] = frames["spatial_fold"]["y_pred"].to_numpy(float)
    correlation_frame["typewell"] = frames["typewell_fold"]["y_pred"].to_numpy(float)
    gates = {
        "standard_nested_gain": standard["nested_rmse_delta"] <= -float(validation.get("minimum_standard_gain", 0.02)),
        "standard_bootstrap": standard["bootstrap"]["ci_97_5"] < 0.0,
        "standard_p90_nonworse": standard["nested_metrics"]["well_rmse_p90"] <= standard["base_metrics"]["well_rmse_p90"] * (1.0 + float(validation.get("tail_tolerance", 0.005))),
        "spatial_nested_nonworse": spatial["nested_rmse_delta"] <= float(validation.get("holdout_tolerance", 0.01)),
        "typewell_nested_nonworse": typewell["nested_rmse_delta"] <= float(validation.get("holdout_tolerance", 0.01)),
        "robust_local_profile": robust_name is not None,
    }
    promoted = all(gates.values())
    summary = {
        "promoted_to_all_train_uncertainty_model": promoted,
        "experiment": "stage13_emission_uncertainty_gate",
        "alignment": alignment,
        "family_reports": family_reports,
        "error_correlations": prediction_error_correlation(correlation_frame, ["standard", "spatial", "typewell"]),
        "robust_local_profile": next((row for row in profiles if row["name"] == robust_name), None),
        "gates": gates,
        "next_step": "Train full-data emission ensemble and calibrate the promoted uncertainty gate." if promoted else "Move from deterministic gating to a cross-fitted learned residual/uncertainty model.",
    }
    write_json(output / "gate_summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {"stage12b_run": str(stage12b), "stage12c_run": str(stage12c), "artifact_dir": str(args.artifact_dir.resolve()), "run_id": args.run_id, "limit_wells": args.limit_wells}
    write_yaml(output / "config.yaml", config)
    print({"promoted_to_all_train_uncertainty_model": promoted, "standard_delta": standard["nested_rmse_delta"], "spatial_delta": spatial["nested_rmse_delta"], "typewell_delta": typewell["nested_rmse_delta"], "bootstrap_95pct": {family: [report["bootstrap"]["ci_2_5"], report["bootstrap"]["ci_97_5"]] for family, report in family_reports.items()}, "error_correlations": summary["error_correlations"], "robust_local_profile": summary["robust_local_profile"], "gates": gates, "next_step": summary["next_step"]}, flush=True)
    print(f"run artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
