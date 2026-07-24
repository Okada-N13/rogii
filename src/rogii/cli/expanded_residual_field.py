from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.branch_retrieval import _bootstrap, _metrics
from rogii.cli.prefix_residual_field import _correction_prediction, residual_features
from rogii.cli.prefix_router import FOLD_FAMILIES, _family_report, build_candidates
from rogii.cli.trajectory_residual import _typewell_folds
from rogii.config import load_config
from rogii.models.spatial_surface_state import anchored_spatial_plane


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 28A expanded strong-base residual field")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--public-oof-run", type=Path, required=True)
    parser.add_argument("--split-manifest-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit-training-cuts", type=int)
    return parser


def expanded_residual_features(
    horizontal: pd.DataFrame,
    cut_index: int,
    candidates: dict[str, np.ndarray],
    base_name: str,
    requested_fraction: float,
    *,
    plane_window_ft: float,
    plane_ridge: float,
) -> tuple[np.ndarray, list[str]]:
    base_features, names = residual_features(
        horizontal, cut_index, candidates, base_name, requested_fraction
    )
    cut = int(cut_index)
    prefix, suffix = horizontal.iloc[:cut], horizontal.iloc[cut:]
    base = np.asarray(candidates[base_name], float)
    plane, gradient = anchored_spatial_plane(
        horizontal, cut, window_ft=plane_window_ft, ridge=plane_ridge
    )
    anchor = prefix.iloc[-1]
    dx = suffix["X"].to_numpy(float) - float(anchor["X"])
    dy = suffix["Y"].to_numpy(float) - float(anchor["Y"])
    distance = np.hypot(dx, dy)
    md = suffix["MD"].to_numpy(float)
    base_u = base + suffix["Z"].to_numpy(float)
    base_u_gradient = np.gradient(base_u, md) * 1000.0 if len(base) > 1 else np.zeros(len(base))
    additions = np.column_stack(
        [
            (plane - base) / 10.0,
            dx / 1000.0,
            dy / 1000.0,
            distance / 1000.0,
            base_u_gradient,
            np.full(len(base), float(gradient[0]) * 1000.0),
            np.full(len(base), float(gradient[1]) * 1000.0),
        ]
    ).astype(np.float32)
    output = np.column_stack([base_features, additions]).astype(np.float32)
    output_names = names + [
        "xy_plane_minus_base_10ft", "x_from_cut_kft", "y_from_cut_kft",
        "xy_from_cut_kft", "base_u_gradient_per_kft",
        "prefix_plane_gx_per_kft", "prefix_plane_gy_per_kft",
    ]
    if not np.isfinite(output).all():
        bad = [output_names[i] for i in np.flatnonzero(~np.isfinite(output).all(axis=0))]
        raise RuntimeError(f"Expanded residual features contain non-finite values: {bad}")
    return output, output_names


def smooth_residual_target(values: np.ndarray, window_rows: int, cap_ft: float) -> np.ndarray:
    residual = pd.Series(np.asarray(values, float))
    smoothed = residual.rolling(
        int(window_rows), center=True, min_periods=1
    ).mean().to_numpy(float)
    return np.clip(smoothed, -float(cap_ft), float(cap_ft)).astype(np.float32)


def _model(config: dict[str, Any], seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=float(config.get("learning_rate", 0.04)),
        max_iter=int(config.get("max_iter", 240)),
        max_leaf_nodes=int(config.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(config.get("min_samples_leaf", 100)),
        l2_regularization=float(config.get("l2_regularization", 6.0)),
        early_stopping=False,
        random_state=int(seed),
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    stage16, stage17 = args.stage16b_run.resolve(), args.stage17a_run.resolve()
    public_run = args.public_oof_run.resolve()
    manifest_run, validation_run = args.split_manifest_run.resolve(), args.validation_run.resolve()
    manifest = json.loads((manifest_run / "summary.json").read_text(encoding="utf-8"))
    if manifest.get("training_wells") != 500 or manifest.get("confirmation_wells") != 120:
        raise AssertionError("Stage 28A requires the frozen 500/120 Stage 24 split")
    if any(manifest.get("overlaps", {}).values()):
        raise AssertionError("Stage 24 split manifest contains well overlap")
    summary17 = json.loads((stage17 / "summary.json").read_text(encoding="utf-8"))
    if summary17.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 17A provenance mismatch")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_parquet(stage17 / "cut_report.parquet")
    training_ids = pd.read_parquet(manifest_run / "training_cut_ids.parquet", columns=["cut_id"])
    validation_ids = pd.read_parquet(
        validation_run / "confidence_cut_report.parquet", columns=["cut_id"]
    )
    training = cuts[cuts["cut_id"].isin(training_ids["cut_id"])].copy().sort_values("cut_id")
    validation = cuts[cuts["cut_id"].isin(validation_ids["cut_id"])].copy().sort_values("cut_id")
    if args.limit_training_cuts is not None:
        training = training.head(int(args.limit_training_cuts)).copy()
    training_wells = set(training["well_id"].astype(str))
    validation_wells = set(validation["well_id"].astype(str))
    overlap = sorted(training_wells & validation_wells)
    if overlap:
        raise AssertionError(f"Training/design-validation overlap: {overlap[:5]}")
    assignments = pd.read_parquet(stage16 / "well_assignments.parquet")
    assignments["well_id"] = assignments["well_id"].astype(str)
    assignments["typewell_fold"] = _typewell_folds(assignments, 5, seed)
    columns = ["well_id", "spatial_fold", "typewell_fold", "branch_group_fold"]
    training = training.merge(assignments[columns], on="well_id", how="left", validate="many_to_one")
    validation = validation.merge(assignments[columns], on="well_id", how="left", validate="many_to_one")
    for frame in (training, validation):
        frame["stage16_fold"] = frame["stage16_fold"].astype(int)
    wells = sorted(training_wells | validation_wells)
    public = pd.read_parquet(
        public_run / "base_oof.parquet",
        columns=["well_id", "row_index", "y_pred"],
        filters=[("well_id", "in", wells)],
    )
    public["well_id"] = public["well_id"].astype(str)
    public_by_well = {well: frame.sort_values("row_index") for well, frame in public.groupby("well_id")}
    train_dir = args.data_dir.resolve() / "train"

    @lru_cache(maxsize=32)
    def load_well(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train_dir / f"{well_id}__horizontal_well.csv")

    @lru_cache(maxsize=32)
    def load_typewell(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train_dir / f"{well_id}__typewell.csv")

    candidate_config = dict(config.get("candidates", {}))
    feature_config = dict(config.get("features", {}))
    model_config = dict(config.get("model", {}))
    base_name = str(model_config.get("base_candidate", "top_pf_a130"))
    stride = int(model_config.get("training_stride", 8))
    target_window = int(model_config.get("target_smoothing_rows", 101))
    target_cap = float(model_config.get("target_cap_ft", 24.0))
    train_parts: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    for position, cut in enumerate(training.itertuples(index=False), 1):
        well_id, outer = str(cut.well_id), int(cut.cut_index)
        horizontal, typewell = load_well(well_id), load_typewell(well_id)
        source = public_by_well[well_id]
        original = int(source["row_index"].min())
        candidates = build_candidates(
            horizontal, typewell, outer,
            source["y_pred"].to_numpy(float)[outer - original :], candidate_config,
        )
        features, names = expanded_residual_features(
            horizontal, outer, candidates, base_name, float(cut.requested_fraction),
            plane_window_ft=float(feature_config.get("plane_window_ft", 1200.0)),
            plane_ridge=float(feature_config.get("plane_ridge", 0.05)),
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise AssertionError("Feature schema changed")
        raw_target = horizontal["TVT"].to_numpy(float)[outer:] - candidates[base_name]
        target = smooth_residual_target(raw_target, target_window, target_cap)
        index = np.arange(0, len(features), stride, dtype=int)
        train_parts.append(
            {
                "fold": int(cut.stage16_fold), "x": features[index], "y": target[index],
                "weight": np.full(len(index), 1.0 / max(len(index), 1), np.float32),
            }
        )
        if position % 25 == 0:
            print(f"expanded residual training features {position}/{len(training)} cuts", flush=True)
    x = np.concatenate([part["x"] for part in train_parts])
    y = np.concatenate([part["y"] for part in train_parts])
    weights = np.concatenate([part["weight"] for part in train_parts])
    folds = np.concatenate(
        [np.full(len(part["y"]), part["fold"], dtype=np.int8) for part in train_parts]
    )
    weights *= len(weights) / weights.sum()
    models = []
    for fold in sorted(np.unique(folds)):
        use = folds != fold
        model = _model(model_config, seed + int(fold))
        model.fit(x[use], y[use], sample_weight=weights[use])
        models.append(model)
        print({"trained_fold_model": int(fold), "rows": int(use.sum())}, flush=True)

    correction = dict(config.get("correction", {}))
    diagnostic_weights = [float(value) for value in correction.get("diagnostic_weights", [0.1, 0.2])]
    primary_weight = float(correction.get("primary_weight", 0.20))
    if primary_weight not in diagnostic_weights:
        diagnostic_weights.append(primary_weight)
    cut_rows = []
    invariance = []
    for position, cut in enumerate(validation.itertuples(index=False), 1):
        well_id, outer = str(cut.well_id), int(cut.cut_index)
        horizontal, typewell = load_well(well_id), load_typewell(well_id)
        source = public_by_well[well_id]
        original = int(source["row_index"].min())
        candidates = build_candidates(
            horizontal, typewell, outer,
            source["y_pred"].to_numpy(float)[outer - original :], candidate_config,
        )
        features, names = expanded_residual_features(
            horizontal, outer, candidates, base_name, float(cut.requested_fraction),
            plane_window_ft=float(feature_config.get("plane_window_ft", 1200.0)),
            plane_ridge=float(feature_config.get("plane_ridge", 0.05)),
        )
        if names != feature_names:
            raise AssertionError("Validation feature schema changed")
        raw_residual = np.mean([model.predict(features) for model in models], axis=0)
        base = candidates[base_name]
        truth = horizontal["TVT"].to_numpy(float)[outer:]
        row: dict[str, Any] = {
            "cut_id": str(cut.cut_id), "well_id": well_id,
            "requested_fraction": float(cut.requested_fraction),
            **{family: int(getattr(cut, family)) for family in FOLD_FAMILIES},
            "suffix_rows": len(truth), "base_sse": float(np.square(base - truth).sum()),
            "raw_residual_mean": float(np.mean(raw_residual)),
            "raw_residual_std": float(np.std(raw_residual)),
        }
        for weight in diagnostic_weights:
            prediction = _correction_prediction(
                base, raw_residual, weight,
                float(correction.get("cap_ft", 8.0)),
                float(correction.get("ramp_rows", 96.0)),
            )
            row[f"candidate_sse_w{int(round(weight * 1000)):03d}"] = float(
                np.square(prediction - truth).sum()
            )
        cut_rows.append(row)
        if len(invariance) < 8:
            changed = horizontal.copy()
            changed.loc[changed.index >= outer, "TVT"] += 9999.0
            changed_candidates = build_candidates(
                changed, typewell, outer,
                source["y_pred"].to_numpy(float)[outer - original :], candidate_config,
            )
            changed_features, _ = expanded_residual_features(
                changed, outer, changed_candidates, base_name, float(cut.requested_fraction),
                plane_window_ft=float(feature_config.get("plane_window_ft", 1200.0)),
                plane_ridge=float(feature_config.get("plane_ridge", 0.05)),
            )
            invariance.append(np.array_equal(features, changed_features))
        if position % 10 == 0:
            print(f"expanded residual validation {position}/{len(validation)} cuts", flush=True)

    report = pd.DataFrame(cut_rows)
    primary_column = f"candidate_sse_w{int(round(primary_weight * 1000)):03d}"
    report["candidate_sse"] = report[primary_column]
    metrics = _metrics(report)
    bootstrap = _bootstrap(report, int(config.get("validation", {}).get("bootstrap_resamples", 4000)), seed)
    family_reports = {family: _family_report(report, family) for family in FOLD_FAMILIES}
    rows = int(report["suffix_rows"].sum())
    base_rmse = float(np.sqrt(report["base_sse"].sum() / rows))
    weight_report = []
    for weight in sorted(diagnostic_weights):
        candidate = float(np.sqrt(report[f"candidate_sse_w{int(round(weight * 1000)):03d}"].sum() / rows))
        weight_report.append({"weight": weight, "base_rmse": base_rmse, "candidate_rmse": candidate, "rmse_delta": candidate - base_rmse})
    validation_config = dict(config.get("validation", {}))
    gates = {
        "hidden_target_invariance": bool(invariance) and all(invariance),
        "training_validation_well_overlap_zero": len(overlap) == 0,
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
    report.to_parquet(output / "expanded_residual_cut_report.parquet", index=False)
    stage_name = str(config.get("stage_name", "stage28a"))
    promotion_key = str(
        config.get("promotion_key", "promoted_to_stage28b_reserved_confirmation")
    )
    summary = {
        f"{stage_name}_complete": True,
        promotion_key: promoted,
        "stage16b_manifest_sha256": expected_hash,
        "training_cuts": len(training), "training_wells": len(training_wells),
        "validation_cuts": len(validation), "validation_wells": len(validation_wells),
        "training_rows": len(x), "feature_count": len(feature_names or []),
        "ensemble_models": len(models), "primary_weight": primary_weight,
        "base_rmse": base_rmse, "candidate_rmse": metrics["candidate_rmse"],
        "rmse_delta": metrics["rmse_delta"], "bootstrap_95pct": bootstrap,
        "cut_rmse_p90_delta": metrics["cut_rmse_p90_delta"],
        "family_reports": family_reports, "weight_report": weight_report,
        "gates": gates, "reserved_confirmation_used": False,
        "next_step": (
            str(config.get("promoted_next_step", "Run exactly one reserved confirmation audit."))
            if promoted
            else str(config.get("rejected_next_step", "Reject this residual experiment and keep the confirmation reserve sealed."))
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16), "stage17a_run": str(stage17),
        "public_oof_run": str(public_run), "split_manifest_run": str(manifest_run),
        "validation_run": str(validation_run), "data_dir": str(args.data_dir.resolve()),
        "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
