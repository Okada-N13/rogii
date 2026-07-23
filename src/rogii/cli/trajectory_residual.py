from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.branch_retrieval import _bootstrap, _metrics
from rogii.config import load_config
from rogii.data.multicut import build_cut_record, feature_columns, feature_schema_hash
from rogii.models.trajectory_residual import (
    COEFFICIENT_COLUMNS,
    apply_residual_coefficients,
    fit_residual_coefficients,
)


FOLD_FAMILIES = ("fold", "spatial_fold", "typewell_fold", "branch_group_fold")
TYPEWELL_COLUMNS = [
    "typewell_gr_mean", "typewell_gr_std", "typewell_gr_q10", "typewell_gr_q50",
    "typewell_gr_q90", "typewell_tvt_min", "typewell_tvt_max", "typewell_tvt_span",
]
EXTRA_FEATURE_COLUMNS = [
    "base_tvt_start", "base_tvt_end", "base_tvt_change", "base_tvt_std",
    "base_u_start", "base_u_end", "base_u_change", "base_u_std",
    "base_boundary_jump", "base_u_slope_per_kft", "base_u_curve",
    "prefix_typewell_gr_rmse", "prefix_typewell_gr_correlation",
    "base_typewell_gr_best_shift", "base_typewell_gr_best_rmse",
    "base_typewell_gr_zero_rmse", "base_typewell_gr_shift_gain",
    "base_typewell_gr_correlation",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 19A cross-fitted low-dimensional trajectory residual")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--stage17b-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit-cuts", type=int)
    return parser


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3 or np.std(left[finite]) < 1e-8 or np.std(right[finite]) < 1e-8:
        return 0.0
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def _typewell_match(
    tvt: np.ndarray,
    horizontal_gr: np.ndarray,
    typewell: pd.DataFrame,
) -> tuple[float, float]:
    tw = typewell[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna().sort_values("TVT")
    tw = tw.groupby("TVT", as_index=False)["GR"].mean()
    if len(tw) < 3:
        return float("nan"), 0.0
    prediction = np.interp(
        np.asarray(tvt, float), tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float),
        left=np.nan, right=np.nan,
    )
    gr = np.asarray(horizontal_gr, float)
    finite = np.isfinite(prediction) & np.isfinite(gr)
    if int(finite.sum()) < 3:
        return float("nan"), 0.0
    rmse = float(np.sqrt(np.mean(np.square(prediction[finite] - gr[finite]))))
    return rmse, _safe_correlation(prediction[finite], gr[finite])


def _cut_feature_record(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cut_index: int,
    base_prediction: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float | int | str]:
    """Build inference-available features; suffix TVT is never read here."""
    record = build_cut_record(
        horizontal, typewell, int(cut_index),
        prefix_window_ft=float(config.get("prefix_window_ft", 800.0)),
        target_ridge=float(config.get("target_ridge", 1.0)),
    )
    for column in ("target_slope_correction", "target_curvature"):
        record.pop(column, None)
    base = np.asarray(base_prediction, float)
    suffix = horizontal.iloc[int(cut_index):]
    if len(base) != len(suffix) or not np.isfinite(base).all():
        raise ValueError("strong-base prediction does not match the pseudo-test suffix")
    z = suffix["Z"].to_numpy(float)
    base_u = base + z
    x = np.linspace(0.0, 1.0, len(base), dtype=float)
    if len(base) >= 3:
        polynomial = np.polyfit(x, base_u - base_u[0], 2)
        base_slope = float(polynomial[1] * 1000.0 / max(float(record["horizon_md_ft"]), 1.0))
        base_curve = float(polynomial[0])
    else:
        base_slope, base_curve = 0.0, 0.0
    prefix = horizontal.iloc[: int(cut_index)]
    prefix_rmse, prefix_corr = _typewell_match(
        prefix["TVT"].to_numpy(float), prefix["GR"].to_numpy(float), typewell
    )
    shifts = [float(value) for value in config.get("typewell_shift_grid_ft", [-30, -20, -10, 0, 10, 20, 30])]
    suffix_gr = suffix["GR"].to_numpy(float)
    matches = [(shift, *_typewell_match(base + shift, suffix_gr, typewell)) for shift in shifts]
    finite_matches = [item for item in matches if np.isfinite(item[1])]
    best_shift, best_rmse, best_corr = min(finite_matches, key=lambda item: (item[1], abs(item[0]))) if finite_matches else (0.0, float("nan"), 0.0)
    zero_rmse = next((item[1] for item in matches if item[0] == 0.0), best_rmse)
    record.update({
        "base_tvt_start": float(base[0]), "base_tvt_end": float(base[-1]),
        "base_tvt_change": float(base[-1] - base[0]), "base_tvt_std": float(np.std(base)),
        "base_u_start": float(base_u[0]), "base_u_end": float(base_u[-1]),
        "base_u_change": float(base_u[-1] - base_u[0]), "base_u_std": float(np.std(base_u)),
        "base_boundary_jump": float(base[0] - prefix["TVT"].iloc[-1]),
        "base_u_slope_per_kft": base_slope, "base_u_curve": base_curve,
        "prefix_typewell_gr_rmse": prefix_rmse,
        "prefix_typewell_gr_correlation": prefix_corr,
        "base_typewell_gr_best_shift": best_shift,
        "base_typewell_gr_best_rmse": best_rmse,
        "base_typewell_gr_zero_rmse": zero_rmse,
        "base_typewell_gr_shift_gain": float(zero_rmse - best_rmse) if np.isfinite(zero_rmse) and np.isfinite(best_rmse) else 0.0,
        "base_typewell_gr_correlation": best_corr,
    })
    return record


def _hidden_target_invariance(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cut_index: int,
    base_prediction: np.ndarray,
    config: dict[str, Any],
) -> bool:
    first = _cut_feature_record(horizontal, typewell, cut_index, base_prediction, config)
    changed = horizontal.copy()
    changed.loc[changed.index[int(cut_index):], "TVT"] += 997.0
    second = _cut_feature_record(changed, typewell, cut_index, base_prediction, config)
    keys = sorted(first)
    left = np.asarray([first[key] for key in keys if key not in {"well_id", "cut_id"}], float)
    right = np.asarray([second[key] for key in keys if key not in {"well_id", "cut_id"}], float)
    return bool(np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True))


def _typewell_folds(assignments: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    columns = [column for column in TYPEWELL_COLUMNS if column in assignments]
    if len(columns) != len(TYPEWELL_COLUMNS):
        raise ValueError(f"Stage 16B assignments are missing typewell signatures: {sorted(set(TYPEWELL_COLUMNS) - set(columns))}")
    count = min(int(n_folds), len(assignments))
    values = StandardScaler().fit_transform(assignments[columns].to_numpy(float))
    return pd.Series(
        KMeans(n_clusters=count, random_state=int(seed), n_init=20).fit_predict(values).astype(np.int16),
        index=assignments.index,
    )


def _model(config: dict[str, Any], seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=float(config.get("learning_rate", 0.04)),
        max_iter=int(config.get("max_iter", 220)),
        max_leaf_nodes=int(config.get("max_leaf_nodes", 15)),
        max_depth=int(config.get("max_depth", 4)),
        min_samples_leaf=int(config.get("min_samples_leaf", 30)),
        l2_regularization=float(config.get("l2_regularization", 4.0)),
        early_stopping=False,
        random_state=int(seed),
    )


def _crossfit_coefficients(
    records: pd.DataFrame,
    feature_names: list[str],
    fold_column: str,
    config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    output = records[["cut_id", "well_id", fold_column]].copy()
    predicted_columns = [column.replace("target_", "pred_") for column in COEFFICIENT_COLUMNS]
    for column in predicted_columns:
        output[column] = np.nan
    report: list[dict[str, Any]] = []
    x = records[feature_names].replace([np.inf, -np.inf], np.nan)
    weights = np.sqrt(records["suffix_rows"].to_numpy(float))
    for fold in sorted(records[fold_column].unique()):
        train = records[fold_column] != fold
        valid = records[fold_column] == fold
        for target_index, (target, prediction) in enumerate(zip(COEFFICIENT_COLUMNS, predicted_columns, strict=True)):
            model = _model(config, seed + int(fold) * 10 + target_index)
            model.fit(x.loc[train], records.loc[train, target], sample_weight=weights[train])
            output.loc[valid, prediction] = model.predict(x.loc[valid])
        report.append({"fold": int(fold), "training_cuts": int(train.sum()), "validation_cuts": int(valid.sum())})
    if output[predicted_columns].isna().any().any():
        raise AssertionError(f"Incomplete {fold_column} coefficient OOF predictions")
    return output, report


def _evaluate(
    records: pd.DataFrame,
    coefficient_oof: pd.DataFrame,
    fold_column: str,
    base_by_cut: Any,
    load_well: Any,
    profile: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    coefficient_map = coefficient_oof.set_index("cut_id")
    rows = []
    predicted_columns = [column.replace("target_", "pred_") for column in COEFFICIENT_COLUMNS]
    for cut in records.sort_values(["well_id", "cut_index"], kind="stable").itertuples(index=False):
        base = base_by_cut.get_group(str(cut.cut_id)).sort_values("row_index", kind="stable")["y_pred"].to_numpy(float)
        truth = load_well(str(cut.well_id))["TVT"].to_numpy(float)[int(cut.cut_index):]
        coefficients = coefficient_map.loc[str(cut.cut_id), predicted_columns].to_numpy(float)
        candidate = apply_residual_coefficients(
            base, coefficients, weight=float(profile["weight"]), cap_ft=float(profile["cap_ft"]),
            ramp_rows=float(profile["ramp_rows"]),
        )
        rows.append({
            "cut_id": str(cut.cut_id), "well_id": str(cut.well_id),
            "stage16_fold": int(getattr(cut, fold_column)),
            "requested_fraction": float(cut.requested_fraction), "suffix_rows": len(truth),
            "base_sse": float(np.square(base - truth).sum()),
            "candidate_sse": float(np.square(candidate - truth).sum()),
        })
    frame = pd.DataFrame.from_records(rows)
    metrics = _metrics(frame)
    base_well = frame.groupby("well_id", sort=True).agg(sse=("base_sse", "sum"), rows=("suffix_rows", "sum"))
    candidate_well = frame.groupby("well_id", sort=True).agg(sse=("candidate_sse", "sum"), rows=("suffix_rows", "sum"))
    base_rmse = np.sqrt(base_well["sse"] / base_well["rows"])
    candidate_rmse = np.sqrt(candidate_well["sse"] / candidate_well["rows"])
    metrics.update({
        "well_rmse_p90_delta": float(candidate_rmse.quantile(0.9) - base_rmse.quantile(0.9)),
        "well_rmse_max_delta": float(candidate_rmse.max() - base_rmse.max()),
        "worst10_sse_share_delta": float(
            candidate_well.nlargest(max(1, int(np.ceil(0.1 * len(candidate_well)))), "sse")["sse"].sum() / candidate_well["sse"].sum()
            - base_well.nlargest(max(1, int(np.ceil(0.1 * len(base_well)))), "sse")["sse"].sum() / base_well["sse"].sum()
        ),
    })
    return frame, metrics


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    stage16, stage17a, stage17b = args.stage16b_run.resolve(), args.stage17a_run.resolve(), args.stage17b_run.resolve()
    stage17b_summary = json.loads((stage17b / "summary.json").read_text(encoding="utf-8"))
    if stage17b_summary.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 17B does not use the frozen Stage 16B manifest")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_parquet(stage17b / "cut_report.parquet")
    cuts = cuts[cuts["evaluation_role"] == "primary"].sort_values(["well_id", "cut_index"], kind="stable").reset_index(drop=True)
    if args.limit_cuts is not None:
        cuts = cuts.head(int(args.limit_cuts)).copy()
    cut_ids = cuts["cut_id"].astype(str).tolist()
    eligible = cuts.loc[cuts["replay_eligible"], "cut_id"].astype(str).tolist()
    uncovered = cuts.loc[~cuts["replay_eligible"], "cut_id"].astype(str).tolist()
    parts = []
    prediction_columns = ["cut_id", "row_index", "y_pred"]
    if eligible:
        parts.append(pd.read_parquet(
            stage17a / "replay_predictions.parquet",
            columns=prediction_columns, filters=[("cut_id", "in", eligible)],
        ))
    if uncovered:
        parts.append(pd.read_parquet(
            stage17b / "selector_predictions.parquet",
            columns=prediction_columns, filters=[("cut_id", "in", uncovered)],
        ))
    base_predictions = pd.concat(parts, ignore_index=True)
    base_predictions["cut_id"] = base_predictions["cut_id"].astype(str)
    if set(base_predictions["cut_id"].unique()) != set(cut_ids):
        raise AssertionError("Strong-base predictions do not cover every selected cut")
    base_by_cut = base_predictions.groupby("cut_id", sort=False)

    assignments = pd.read_parquet(stage16 / "well_assignments.parquet")
    assignments["well_id"] = assignments["well_id"].astype(str)
    assignments["typewell_fold"] = _typewell_folds(
        assignments, int(config.get("validation", {}).get("n_typewell_folds", 5)), seed
    )
    fold_columns = ["well_id", *FOLD_FAMILIES]
    cuts = cuts.merge(assignments[fold_columns], on="well_id", how="left", validate="many_to_one")
    data_train = args.data_dir.resolve() / "train"

    @lru_cache(maxsize=64)
    def load_well(well_id: str) -> pd.DataFrame:
        frame = pd.read_csv(data_train / f"{well_id}__horizontal_well.csv")
        frame["well_id"] = str(well_id)
        frame["row_index"] = np.arange(len(frame), dtype=np.int64)
        return frame

    @lru_cache(maxsize=64)
    def load_typewell(well_id: str) -> pd.DataFrame:
        return pd.read_csv(data_train / f"{well_id}__typewell.csv")

    feature_config = dict(config.get("features", {}))
    target_config = dict(config.get("target", {}))
    records = []
    invariance_checks = []
    for position, cut in enumerate(cuts.itertuples(index=False), 1):
        well_id, cut_id, cut_index = str(cut.well_id), str(cut.cut_id), int(cut.cut_index)
        horizontal, typewell = load_well(well_id), load_typewell(well_id)
        base = base_by_cut.get_group(cut_id).sort_values("row_index", kind="stable")["y_pred"].to_numpy(float)
        feature = _cut_feature_record(horizontal, typewell, cut_index, base, feature_config)
        truth = horizontal["TVT"].to_numpy(float)[cut_index:]
        coefficients = fit_residual_coefficients(
            base, truth, ramp_rows=float(target_config.get("ramp_rows", 96.0)),
            ridge=float(target_config.get("ridge", 1.0)),
        )
        records.append({
            **feature, "cut_id": cut_id, "well_id": well_id,
            "requested_fraction": float(cut.requested_fraction), "evaluation_role": str(cut.evaluation_role),
            **{column: float(value) for column, value in zip(COEFFICIENT_COLUMNS, coefficients, strict=True)},
            **{family: int(getattr(cut, family)) for family in FOLD_FAMILIES},
        })
        if len(invariance_checks) < 8:
            invariance_checks.append(_hidden_target_invariance(horizontal, typewell, cut_index, base, feature_config))
        if position % 200 == 0:
            print(f"trajectory features {position}/{len(cuts)} cuts", flush=True)
    records = pd.DataFrame.from_records(records)
    hidden_target_invariance = bool(invariance_checks) and all(invariance_checks)
    if not hidden_target_invariance:
        raise AssertionError("Stage 19A hidden-target invariance failed")
    excluded = set(COEFFICIENT_COLUMNS) | {"evaluation_role", *FOLD_FAMILIES}
    features = [column for column in feature_columns(records) if column not in excluded]
    missing_extras = sorted(set(EXTRA_FEATURE_COLUMNS) - set(features))
    if missing_extras:
        raise AssertionError(f"Stage 19A feature construction is incomplete: {missing_extras}")

    profile = dict(config.get("profile", {}))
    family_reports, model_reports = {}, {}
    standard_frame = None
    for family_index, family in enumerate(FOLD_FAMILIES):
        print(f"cross-fitting trajectory coefficients: {family}", flush=True)
        coefficient_oof, model_report = _crossfit_coefficients(
            records, features, family, dict(config.get("model", {})), seed + family_index * 100
        )
        coefficient_oof.to_parquet(output / f"{family}_coefficient_oof.parquet", index=False)
        evaluation, metrics = _evaluate(records, coefficient_oof, family, base_by_cut, load_well, profile)
        family_reports[family], model_reports[family] = metrics, model_report
        if family == "fold":
            standard_frame = evaluation
            evaluation.to_parquet(output / "standard_cut_metrics.parquet", index=False)
    assert standard_frame is not None

    diagnostic_rows = []
    standard_coefficients = pd.read_parquet(output / "fold_coefficient_oof.parquet")
    for weight in [float(value) for value in config.get("diagnostics", {}).get("weights", [0.25, 0.5, 0.75, 1.0])]:
        for cap in [float(value) for value in config.get("diagnostics", {}).get("caps_ft", [8, 12, 16, 24])]:
            candidate_profile = {**profile, "weight": weight, "cap_ft": cap}
            _, metrics = _evaluate(records, standard_coefficients, "fold", base_by_cut, load_well, candidate_profile)
            diagnostic_rows.append({"weight": weight, "cap_ft": cap, **metrics})
    pd.DataFrame.from_records(diagnostic_rows).to_parquet(output / "profile_report.parquet", index=False)

    validation = dict(config.get("validation", {}))
    standard = family_reports["fold"]
    bootstrap = _bootstrap(standard_frame, int(validation.get("bootstrap_resamples", 3000)), seed)
    family_consistency = {
        family: report["improved_folds"] >= int(np.ceil(len(report["fold_report"]) * float(validation.get("minimum_improved_fold_fraction", 0.8))))
        for family, report in family_reports.items()
    }
    gates = {
        "hidden_target_invariance": hidden_target_invariance,
        "standard_gain": standard["rmse_delta"] <= -float(validation.get("minimum_standard_gain", 0.30)),
        "standard_bootstrap": bootstrap[1] < 0.0,
        "standard_fold_consistency": family_consistency["fold"],
        "spatial_gain": family_reports["spatial_fold"]["rmse_delta"] < 0.0,
        "spatial_fold_consistency": family_consistency["spatial_fold"],
        "typewell_gain": family_reports["typewell_fold"]["rmse_delta"] < 0.0,
        "typewell_fold_consistency": family_consistency["typewell_fold"],
        "branch_group_gain": family_reports["branch_group_fold"]["rmse_delta"] < 0.0,
        "branch_group_fold_consistency": family_consistency["branch_group_fold"],
        "well_p90_nonworse": standard["well_rmse_p90_delta"] <= 0.0,
        "worst10_share_nonworse": standard["worst10_sse_share_delta"] <= 0.0,
    }
    summary = {
        "stage19a_complete": True, "promoted_to_stage19b": bool(all(gates.values())),
        "stage16b_manifest_sha256": expected_hash, "cuts": len(records),
        "wells": int(records["well_id"].nunique()), "feature_count": len(features),
        "feature_columns": features, "feature_schema_hash": feature_schema_hash(features),
        "coefficient_columns": list(COEFFICIENT_COLUMNS), "profile": profile,
        "hidden_target_invariance": hidden_target_invariance,
        "family_reports": family_reports, "model_reports": model_reports,
        "bootstrap_95pct": bootstrap, "gates": gates,
        "runtime_contract": {"donor_search": False, "rowwise_neural_model": False, "predicted_values_per_well": 3},
        "next_step": (
            "Train all-data Stage 19B models and benchmark independent hidden-style inference runtime."
            if all(gates.values()) else
            "Revise the target-free coefficient features before building a Kaggle package."
        ),
    }
    records.to_parquet(output / "cut_features.parquet", index=False)
    assignments.to_parquet(output / "well_assignments.parquet", index=False)
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16), "stage17a_run": str(stage17a), "stage17b_run": str(stage17b),
        "data_dir": str(args.data_dir.resolve()), "run_id": args.run_id, "limit_cuts": args.limit_cuts,
    }
    write_yaml(output / "config.yaml", config)
    print({
        "stage19a_complete": True, "promoted_to_stage19b": summary["promoted_to_stage19b"],
        "cuts": summary["cuts"], "wells": summary["wells"],
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
