from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.branch_retrieval import _bootstrap
from rogii.cli.selector_replay import likelihood_selector
from rogii.cli.selector_resolution import stable_stratified_sample
from rogii.cli.trajectory_residual import (
    EXTRA_FEATURE_COLUMNS,
    FOLD_FAMILIES,
    _crossfit_coefficients,
    _evaluate,
    _hidden_target_invariance,
    _typewell_folds,
    _cut_feature_record,
)
from rogii.config import load_config
from rogii.data.multicut import feature_columns, feature_schema_hash
from rogii.models.trajectory_residual import COEFFICIENT_COLUMNS, fit_residual_coefficients


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 20A top-PF-aligned residual screen")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit-cuts", type=int)
    return parser


def robust_projection(
    horizontal: pd.DataFrame,
    cut_index: int,
    prediction: np.ndarray,
    *,
    degree: int = 3,
    blend_weight: float = 0.75,
) -> np.ndarray:
    """The deterministic SP45 U-space projection used before the final blend."""
    prefix, suffix = horizontal.iloc[:cut_index], horizontal.iloc[cut_index:]
    prediction = np.asarray(prediction, float)
    if len(prediction) != len(suffix):
        raise ValueError("projection prediction length mismatch")
    anchor = float(prefix["TVT"].iloc[-1] + prefix["Z"].iloc[-1])
    md = suffix["MD"].to_numpy(float)
    z = suffix["Z"].to_numpy(float)
    scale = max(float(horizontal["MD"].iloc[-1] - prefix["MD"].iloc[-1]), 1e-6)
    x = (md - float(prefix["MD"].iloc[-1])) / scale
    y = prediction + z - anchor
    if len(x) < int(degree) + 2:
        return prediction.copy()
    coefficients = np.polyfit(x, y, int(degree))
    for _ in range(4):
        residual = y - np.polyval(coefficients, x)
        robust_scale = np.median(np.abs(residual)) * 1.4826 + 1e-6
        weights = 1.0 / (1.0 + np.square(residual / (2.0 * robust_scale)))
        coefficients = np.polyfit(x, y, int(degree), w=weights)
    fitted = anchor + np.polyval(coefficients, x) - z
    output = (1.0 - float(blend_weight)) * prediction + float(blend_weight) * fitted
    if not np.isfinite(output).all():
        raise RuntimeError("SP45 projection produced non-finite values")
    return output


def top_pf_proxy(
    horizontal: pd.DataFrame,
    cut_index: int,
    public_prediction: np.ndarray,
    selector_prediction: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Target-safe approximation of the 470 blend before adaptive overlays."""
    public_prediction = np.asarray(public_prediction, float)
    selector_prediction = np.asarray(selector_prediction, float)
    sp45 = (
        float(config.get("ridge_weight", 0.30)) * public_prediction
        + float(config.get("selector_weight", 0.70)) * selector_prediction
    )
    projected = robust_projection(
        horizontal, cut_index, sp45,
        degree=int(config.get("projection_degree", 3)),
        blend_weight=float(config.get("projection_blend_weight", 0.75)),
    )
    sp45_weight = float(config.get("final_sp45_weight", 0.60))
    output = sp45_weight * projected + (1.0 - sp45_weight) * public_prediction
    if not np.isfinite(output).all():
        raise RuntimeError("top-PF proxy produced non-finite values")
    return output


def _rmse(frame: pd.DataFrame, column: str) -> float:
    return float(np.sqrt(frame[column].sum() / frame["suffix_rows"].sum()))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    stage16, stage17a = args.stage16b_run.resolve(), args.stage17a_run.resolve()
    summary17 = json.loads((stage17a / "summary.json").read_text(encoding="utf-8"))
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    if summary17.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 17A does not use the frozen Stage 16B manifest")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_parquet(stage17a / "cut_report.parquet")
    eligible = cuts[(cuts["evaluation_role"] == "primary") & cuts["replay_eligible"]].copy()
    selected = stable_stratified_sample(
        eligible, int(config.get("sample", {}).get("cuts_per_stratum", 8))
    )
    if args.limit_cuts is not None:
        selected = selected.head(int(args.limit_cuts)).copy()
    selected = selected.sort_values(["well_id", "cut_index"], kind="stable").reset_index(drop=True)
    cut_ids = selected["cut_id"].astype(str).tolist()
    public = pd.read_parquet(
        stage17a / "replay_predictions.parquet",
        columns=["cut_id", "row_index", "y_pred"],
        filters=[("cut_id", "in", cut_ids)],
    )
    public["cut_id"] = public["cut_id"].astype(str)
    public_by_cut = public.groupby("cut_id", sort=False)
    if set(public["cut_id"].unique()) != set(cut_ids):
        raise AssertionError("Public OOF does not cover every selected cut")

    assignments = pd.read_parquet(stage16 / "well_assignments.parquet")
    assignments["well_id"] = assignments["well_id"].astype(str)
    assignments["typewell_fold"] = _typewell_folds(
        assignments, int(config.get("validation", {}).get("n_typewell_folds", 5)), seed
    )
    selected = selected.merge(
        assignments[["well_id", *FOLD_FAMILIES]], on="well_id", how="left", validate="many_to_one"
    )
    train = args.data_dir.resolve() / "train"

    @lru_cache(maxsize=64)
    def load_well(well_id: str) -> pd.DataFrame:
        frame = pd.read_csv(train / f"{well_id}__horizontal_well.csv")
        frame["well_id"] = well_id
        frame["row_index"] = np.arange(len(frame), dtype=np.int64)
        return frame

    @lru_cache(maxsize=64)
    def load_typewell(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train / f"{well_id}__typewell.csv")

    selector_config = dict(config.get("selector", {}))
    blend_config = dict(config.get("top_pf_proxy", {}))
    feature_config = dict(config.get("features", {}))
    target_config = dict(config.get("target", {}))
    records, prediction_parts, comparison_rows, invariance = [], [], [], []
    for position, cut in enumerate(selected.itertuples(index=False), 1):
        well_id, cut_id, cut_index = str(cut.well_id), str(cut.cut_id), int(cut.cut_index)
        horizontal, typewell = load_well(well_id), load_typewell(well_id)
        public_prediction = public_by_cut.get_group(cut_id).sort_values("row_index")["y_pred"].to_numpy(float)
        masked = horizontal[["MD", "Z", "GR", "TVT"]].copy()
        masked.loc[masked.index >= cut_index, "TVT"] = np.nan
        selector, selector_audit = likelihood_selector(masked, typewell, cut_index, selector_config)
        base = top_pf_proxy(horizontal, cut_index, public_prediction, selector, blend_config)
        truth = horizontal["TVT"].to_numpy(float)[cut_index:]
        feature = _cut_feature_record(horizontal, typewell, cut_index, base, feature_config)
        coefficients = fit_residual_coefficients(
            base, truth, ramp_rows=float(target_config.get("ramp_rows", 96.0)),
            ridge=float(target_config.get("ridge", 1.0)),
        )
        records.append({
            **feature, "cut_id": cut_id, "well_id": well_id,
            "requested_fraction": float(cut.requested_fraction), "evaluation_role": "primary",
            **{name: float(value) for name, value in zip(COEFFICIENT_COLUMNS, coefficients, strict=True)},
            **{family: int(getattr(cut, family)) for family in FOLD_FAMILIES},
        })
        prediction_parts.append(pd.DataFrame({
            "cut_id": cut_id, "row_index": np.arange(cut_index, len(horizontal), dtype=np.int32),
            "y_pred": base,
        }))
        comparison_rows.append({
            "cut_id": cut_id, "well_id": well_id, "fold": int(cut.fold),
            "requested_fraction": float(cut.requested_fraction), "suffix_rows": len(truth),
            "public_sse": float(np.square(public_prediction-truth).sum()),
            "selector_sse": float(np.square(selector-truth).sum()),
            "top_pf_proxy_sse": float(np.square(base-truth).sum()),
            **selector_audit,
        })
        if len(invariance) < 12:
            invariance.append(_hidden_target_invariance(horizontal, typewell, cut_index, base, feature_config))
        if position % 25 == 0:
            print(f"top-PF alignment replay {position}/{len(selected)} cuts", flush=True)
    records = pd.DataFrame.from_records(records)
    base_predictions = pd.concat(prediction_parts, ignore_index=True)
    base_by_cut = base_predictions.groupby("cut_id", sort=False)
    comparison = pd.DataFrame.from_records(comparison_rows)
    comparison.to_parquet(output / "base_comparison.parquet", index=False)
    base_predictions.to_parquet(output / "top_pf_proxy_predictions.parquet", index=False)

    excluded = set(COEFFICIENT_COLUMNS) | {"evaluation_role", *FOLD_FAMILIES}
    features = [column for column in feature_columns(records) if column not in excluded]
    if sorted(set(EXTRA_FEATURE_COLUMNS) - set(features)):
        raise AssertionError("Stage 20A feature construction is incomplete")
    profile = dict(config.get("profile", {}))
    family_reports, standard_frame = {}, None
    for family_index, family in enumerate(FOLD_FAMILIES):
        coefficient_oof, model_report = _crossfit_coefficients(
            records, features, family, dict(config.get("model", {})), seed + family_index * 100
        )
        coefficient_oof.to_parquet(output / f"{family}_coefficient_oof.parquet", index=False)
        evaluation, metrics = _evaluate(records, coefficient_oof, family, base_by_cut, load_well, profile)
        family_reports[family] = metrics
        if family == "fold":
            standard_frame = evaluation
            evaluation.to_parquet(output / "standard_cut_metrics.parquet", index=False)
    assert standard_frame is not None

    diagnostics = []
    standard_coefficients = pd.read_parquet(output / "fold_coefficient_oof.parquet")
    for weight in [float(value) for value in config.get("diagnostics", {}).get("weights", [.05,.1,.2,.35,.5])]:
        candidate_profile = {**profile, "weight": weight}
        _, metrics = _evaluate(records, standard_coefficients, "fold", base_by_cut, load_well, candidate_profile)
        diagnostics.append({"weight": weight, **metrics})
    pd.DataFrame.from_records(diagnostics).to_parquet(output / "weight_report.parquet", index=False)

    validation = dict(config.get("validation", {}))
    standard = family_reports["fold"]
    bootstrap = _bootstrap(standard_frame, int(validation.get("bootstrap_resamples", 2000)), seed)
    minimum_fraction = float(validation.get("minimum_improved_fold_fraction", .8))
    family_consistency = {
        family: report["improved_folds"] >= int(np.ceil(len(report["fold_report"])*minimum_fraction))
        for family, report in family_reports.items()
    }
    gates = {
        "hidden_target_invariance": bool(invariance) and all(invariance),
        "public_oof_target_safe": True,
        "top_pf_proxy_constructed": bool(np.isfinite(base_predictions["y_pred"]).all()),
        "standard_gain": standard["rmse_delta"] <= -float(validation.get("minimum_standard_gain", .05)),
        "standard_bootstrap": bootstrap[1] < 0,
        "standard_fold_consistency": family_consistency["fold"],
        "spatial_gain": family_reports["spatial_fold"]["rmse_delta"] < 0,
        "typewell_gain": family_reports["typewell_fold"]["rmse_delta"] < 0,
        "branch_group_gain": family_reports["branch_group_fold"]["rmse_delta"] < 0,
        "well_p90_nonworse": standard["well_rmse_p90_delta"] <= 0,
    }
    summary = {
        "stage20a_complete": True, "promoted_to_stage20b": bool(all(gates.values())),
        "stage16b_manifest_sha256": expected_hash, "sample_cuts": len(records),
        "sample_wells": int(records["well_id"].nunique()), "feature_count": len(features),
        "feature_columns": features, "feature_schema_hash": feature_schema_hash(features),
        "top_pf_proxy_limitations": [
            "public OOF substitutes for the unavailable fold-safe learned 470 branch",
            "visible-prefix candidate selection, tiny model-package overlay, and seed-branch hedge are omitted",
        ],
        "base_comparison": {
            "public_rmse": _rmse(comparison, "public_sse"),
            "selector_rmse": _rmse(comparison, "selector_sse"),
            "top_pf_proxy_rmse": _rmse(comparison, "top_pf_proxy_sse"),
        },
        "profile": profile, "standard_report": standard, "family_reports": family_reports,
        "bootstrap_95pct": bootstrap, "gates": gates,
        "next_step": (
            "Run the aligned residual on all eligible cuts and a separate short-prefix proxy."
            if all(gates.values()) else
            "Do not submit another trajectory residual; redesign the base-aligned target or model."
        ),
    }
    records.to_parquet(output / "cut_features.parquet", index=False)
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16), "stage17a_run": str(stage17a),
        "data_dir": str(args.data_dir.resolve()), "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print({
        "stage20a_complete": True, "promoted_to_stage20b": summary["promoted_to_stage20b"],
        "sample_cuts": summary["sample_cuts"], "sample_wells": summary["sample_wells"],
        "base_comparison": summary["base_comparison"],
        "standard_base_rmse": standard["base_rmse"],
        "standard_candidate_rmse": standard["candidate_rmse"],
        "standard_delta": standard["rmse_delta"],
        "spatial_delta": family_reports["spatial_fold"]["rmse_delta"],
        "typewell_delta": family_reports["typewell_fold"]["rmse_delta"],
        "branch_group_delta": family_reports["branch_group_fold"]["rmse_delta"],
        "bootstrap_95pct": bootstrap, "gates": gates, "next_step": summary["next_step"],
    }, flush=True)


if __name__ == "__main__":
    main()
