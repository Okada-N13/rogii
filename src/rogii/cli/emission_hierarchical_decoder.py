from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.branch_retrieval import _bootstrap, _metrics
from rogii.cli.prefix_router import FOLD_FAMILIES, _family_report
from rogii.config import load_config


FEATURE_COLUMNS = [f"summary_{index}" for index in range(9)]


def _apply_correction(
    surface: np.ndarray,
    correction: np.ndarray,
    cap_ft: float,
    ramp_rows: float,
) -> np.ndarray:
    step = np.arange(1, len(surface) + 1, dtype=float)
    ramp = 1.0 - np.exp(-step / max(float(ramp_rows), 1.0))
    return np.asarray(surface, float) + np.clip(
        ramp * np.asarray(correction, float), -float(cap_ft), float(cap_ft)
    )


def _cut_report(frame: pd.DataFrame, prediction: str) -> pd.DataFrame:
    rows = []
    for cut_id, group in frame.groupby("cut_id", sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "cut_id": cut_id,
                "well_id": str(first["well_id"]),
                "requested_fraction": float(first["requested_fraction"]),
                **{family: int(first[family]) for family in FOLD_FAMILIES},
                "suffix_rows": len(group),
                "base_sse": float(np.square(group["surface"] - group["truth"]).sum()),
                "candidate_sse": float(np.square(group[prediction] - group["truth"]).sum()),
            }
        )
    return pd.DataFrame.from_records(rows)


def _p90_delta(report: pd.DataFrame) -> float:
    base = report.groupby("well_id").agg(sse=("base_sse", "sum"), rows=("suffix_rows", "sum"))
    candidate = report.groupby("well_id").agg(
        sse=("candidate_sse", "sum"), rows=("suffix_rows", "sum")
    )
    return float(
        np.sqrt(candidate.sse / candidate.rows).quantile(0.9)
        - np.sqrt(base.sse / base.rows).quantile(0.9)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 23D hierarchical emission decoder")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage23c-run", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _balanced_weight(frame: pd.DataFrame) -> np.ndarray:
    return 1.0 / frame.groupby("cut_id")["cut_id"].transform("size").to_numpy(float)


def _positive_probability(model: LogisticRegression | None, constant: float, values: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(len(values), float(constant), dtype=float)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(values), dtype=float)
    return model.predict_proba(values)[:, classes.index(1)]


def fit_hierarchical_decoder(
    frame: pd.DataFrame,
    profile: dict[str, Any],
) -> dict[str, Any]:
    features = frame[FEATURE_COLUMNS].to_numpy(float)
    weight = _balanced_weight(frame)
    scaler = StandardScaler().fit(features, sample_weight=weight)
    transformed = scaler.transform(features)
    target = frame["true_offset"].to_numpy(float)
    moving = np.abs(target) >= float(profile["move_threshold_ft"])
    direction = target > 0

    def classifier(labels: np.ndarray, mask: np.ndarray) -> tuple[LogisticRegression | None, float]:
        selected = labels[mask].astype(int)
        selected_weight = weight[mask]
        if len(np.unique(selected)) < 2:
            return None, float(np.average(selected, weights=selected_weight))
        model = LogisticRegression(
            C=float(profile["classifier_c"]),
            solver="lbfgs",
            max_iter=1000,
            random_state=0,
        )
        model.fit(transformed[mask], selected, sample_weight=selected_weight)
        return model, float(np.average(selected, weights=selected_weight))

    move_model, move_constant = classifier(moving, np.ones(len(frame), dtype=bool))
    direction_model, direction_constant = classifier(direction, moving)
    magnitude = Ridge(alpha=float(profile["magnitude_alpha"]))
    magnitude.fit(
        transformed[moving],
        np.abs(target[moving]),
        sample_weight=weight[moving],
    )
    return {
        "scaler": scaler,
        "move_model": move_model,
        "move_constant": move_constant,
        "direction_model": direction_model,
        "direction_constant": direction_constant,
        "magnitude_model": magnitude,
    }


def predict_hierarchical_decoder(
    fitted: dict[str, Any],
    frame: pd.DataFrame,
    profile: dict[str, Any],
) -> np.ndarray:
    transformed = fitted["scaler"].transform(frame[FEATURE_COLUMNS].to_numpy(float))
    move = _positive_probability(
        fitted["move_model"], fitted["move_constant"], transformed
    )
    positive = _positive_probability(
        fitted["direction_model"], fitted["direction_constant"], transformed
    )
    direction = 2.0 * positive - 1.0
    magnitude = np.clip(
        fitted["magnitude_model"].predict(transformed),
        float(profile["move_threshold_ft"]),
        float(profile["magnitude_cap_ft"]),
    )
    move_strength = np.power(np.clip(move, 0.0, 1.0), float(profile["move_power"]))
    direction_strength = np.sign(direction) * np.power(
        np.abs(direction), float(profile["direction_power"])
    )
    return (
        float(profile["decoder_weight"])
        * move_strength
        * direction_strength
        * magnitude
    )


def _apply_by_cut(
    frame: pd.DataFrame,
    correction: np.ndarray,
    prediction_column: str,
    correction_config: dict[str, Any],
) -> pd.DataFrame:
    result = frame.copy()
    result["_correction"] = np.asarray(correction, float)
    parts = []
    for _, group in result.groupby("cut_id", sort=False):
        values = _apply_correction(
            group["surface"].to_numpy(float),
            group["_correction"].to_numpy(float),
            float(correction_config.get("cap_ft", 12.0)),
            float(correction_config.get("ramp_rows", 64.0)),
        )
        parts.append(pd.Series(values, index=group.index))
    result[prediction_column] = pd.concat(parts).sort_index()
    return result.drop(columns="_correction")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    stage23c = args.stage23c_run.resolve()
    summary23c = json.loads((stage23c / "summary.json").read_text(encoding="utf-8"))
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    if summary23c.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 23C manifest provenance mismatch")
    if summary23c.get("promoted_to_stage23d") is not False:
        raise AssertionError("Stage 23D is the repair path for rejected Stage 23C")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    training = pd.read_parquet(stage23c / "training_oof_decoder_rows.parquet")
    validation = pd.read_parquet(stage23c / "validation_decoder_rows.parquet")
    overlap = sorted(
        set(training["well_id"].astype(str)).intersection(validation["well_id"].astype(str))
    )
    if overlap:
        raise AssertionError(f"Training/validation well leakage: {overlap[:5]}")
    required = {
        "cut_id", "well_id", "stage16_fold", "surface", "truth", "true_offset",
        *FOLD_FAMILIES, *FEATURE_COLUMNS,
    }
    for name, frame in (("training", training), ("validation", validation)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise KeyError(f"{name} decoder rows missing columns: {missing}")
        if not np.isfinite(frame[list(FEATURE_COLUMNS) + ["surface", "truth", "true_offset"]]).all().all():
            raise RuntimeError(f"{name} decoder rows contain non-finite values")

    folds = sorted(training["stage16_fold"].astype(int).unique())
    profiles = [dict(value) for value in config.get("decoder_profiles", [])]
    correction_config = dict(config.get("correction", {}))
    selection_config = dict(config.get("selection", {}))
    nested_reports: list[dict[str, Any]] = []
    for profile in profiles:
        name = str(profile["name"])
        outer_parts = []
        for fold in folds:
            inner = training[training["stage16_fold"] != fold]
            outer = training[training["stage16_fold"] == fold]
            fitted = fit_hierarchical_decoder(inner, profile)
            correction = predict_hierarchical_decoder(fitted, outer, profile)
            outer_parts.append(
                _apply_by_cut(
                    outer, correction, f"prediction_{name}", correction_config
                )
            )
        nested = pd.concat(outer_parts).sort_index()
        report = _cut_report(nested, f"prediction_{name}")
        metrics = _metrics(report)
        bootstrap = _bootstrap(
            report, int(selection_config.get("bootstrap_resamples", 2000)), seed
        )
        p90 = _p90_delta(report)
        worst_fold = max(row["delta"] for row in metrics["fold_report"])
        eligible = (
            metrics["rmse_delta"] <= -float(selection_config.get("minimum_gain", 0.1))
            and metrics["improved_folds"] >= int(selection_config.get("minimum_improved_folds", 4))
            and worst_fold <= float(selection_config.get("maximum_worst_fold_delta", 0.05))
            and bootstrap[1] < 0
            and (not bool(selection_config.get("require_p90_nonworse", True)) or p90 <= 0)
        )
        nested_reports.append(
            {
                "profile": name,
                "rmse": metrics["candidate_rmse"],
                "rmse_delta": metrics["rmse_delta"],
                "improved_folds": metrics["improved_folds"],
                "worst_fold_delta": worst_fold,
                "p90_delta": p90,
                "bootstrap_95pct": bootstrap,
                "eligible": eligible,
            }
        )

    eligible = [row for row in nested_reports if row["eligible"]]
    selected_name = (
        min(eligible, key=lambda row: (row["rmse"], row["profile"]))["profile"]
        if eligible else None
    )
    selected_profile = next(
        (profile for profile in profiles if profile["name"] == selected_name), None
    )
    if selected_profile is None:
        validation["prediction"] = validation["surface"]
    else:
        fitted = fit_hierarchical_decoder(training, selected_profile)
        correction = predict_hierarchical_decoder(fitted, validation, selected_profile)
        validation = _apply_by_cut(
            validation, correction, "prediction", correction_config
        )

    validation_report = _cut_report(validation, "prediction")
    metrics = _metrics(validation_report)
    validation_config = dict(config.get("validation", {}))
    bootstrap = _bootstrap(
        validation_report,
        int(validation_config.get("bootstrap_resamples", 3000)),
        seed,
    )
    p90 = _p90_delta(validation_report)
    family_reports = {
        family: _family_report(validation_report, family) for family in FOLD_FAMILIES
    }
    gates = {
        "training_validation_well_overlap_zero": not overlap,
        "training_oof_profile_available": selected_profile is not None,
        "design_validation_gain": metrics["rmse_delta"]
        <= -float(validation_config.get("minimum_gain", 0.1)),
        "design_validation_bootstrap": bootstrap[1] < 0,
        "design_validation_p90_nonworse": p90 <= 0,
        "standard_consistency": metrics["improved_folds"]
        >= int(validation_config.get("minimum_improved_folds", 4)),
        "fraction_consistency": metrics["improved_fractions"]
        >= int(validation_config.get("minimum_improved_fractions", 3)),
        "spatial_consistency": family_reports["spatial_fold"]["improved_folds"]
        >= int(validation_config.get("minimum_spatial_folds", 4)),
        "typewell_consistency": family_reports["typewell_fold"]["improved_folds"]
        >= int(validation_config.get("minimum_typewell_folds", 4)),
        "branch_consistency": family_reports["branch_group_fold"]["improved_folds"]
        >= int(validation_config.get("minimum_branch_folds", 4)),
    }
    promoted = bool(all(gates.values()))
    validation.to_parquet(output / "validation_hierarchical_rows.parquet", index=False)
    validation_report.to_parquet(output / "validation_cut_report.parquet", index=False)
    summary = {
        "stage23d_complete": True,
        "promoted_to_stage23e_disjoint_confirmation": promoted,
        "stage16b_manifest_sha256": expected_hash,
        "training_cuts": int(training["cut_id"].nunique()),
        "training_wells": int(training["well_id"].nunique()),
        "validation_cuts": int(validation["cut_id"].nunique()),
        "validation_wells": int(validation["well_id"].nunique()),
        "training_validation_well_overlap": overlap,
        "nested_profile_report": nested_reports,
        "selected_profile": selected_profile,
        "validation_base_rmse": metrics["base_rmse"],
        "validation_candidate_rmse": metrics["candidate_rmse"],
        "validation_delta": metrics["rmse_delta"],
        "validation_p90_delta": p90,
        "validation_bootstrap_95pct": bootstrap,
        "validation_metrics": metrics,
        "validation_family_reports": family_reports,
        "gates": gates,
        "validation_role": "design_validation_reused_after_stage23b_and_stage23c",
        "next_step": (
            "Run a genuinely disjoint well confirmation; do not package or submit yet."
            if promoted
            else "Reject the hierarchical decoder and redesign the emission target."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage23c_run": str(stage23c),
        "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
