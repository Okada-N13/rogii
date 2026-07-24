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
from rogii.cli.prefix_router import FOLD_FAMILIES, _family_report, build_candidates
from rogii.cli.trajectory_residual import _typewell_folds
from rogii.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 22A disjoint candidate residual field")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--public-oof-run", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit-training-cuts", type=int)
    parser.add_argument("--limit-validation-cuts", type=int)
    return parser


def residual_features(
    horizontal: pd.DataFrame,
    cut_index: int,
    candidates: dict[str, np.ndarray],
    base_name: str,
    requested_fraction: float,
) -> tuple[np.ndarray, list[str]]:
    """Build rowwise features using only visible prefix targets and target-free suffix inputs."""
    cut_index = int(cut_index)
    prefix, suffix = horizontal.iloc[:cut_index], horizontal.iloc[cut_index:]
    base = np.asarray(candidates[base_name], float)
    expected = len(suffix)
    if len(base) != expected:
        raise ValueError("base prediction length mismatch")
    md = suffix["MD"].to_numpy(float)
    gr = suffix["GR"].to_numpy(float)
    z = suffix["Z"].to_numpy(float)
    prefix_gr = prefix["GR"].to_numpy(float)
    prefix_u = prefix["TVT"].to_numpy(float) + prefix["Z"].to_numpy(float)
    prefix_md = prefix["MD"].to_numpy(float)
    origin = float(md[0])
    span = max(float(md[-1] - origin), 1.0)
    gr_scale = max(float(np.std(prefix_gr)), 1.0)
    u_scale = max(float(prefix_md[-1] - prefix_md[0]), 1.0)
    u_slope = float((prefix_u[-1] - prefix_u[0]) / u_scale)
    base_gradient = np.gradient(base, md) if expected > 1 else np.zeros(expected)
    values: list[np.ndarray] = [
        (md - origin) / span,
        (md - origin) / 1000.0,
        (gr - float(np.mean(prefix_gr))) / gr_scale,
        (gr - float(prefix_gr[-1])) / gr_scale,
        (z - float(prefix["Z"].iloc[-1])) / 1000.0,
        (base - float(base[0])) / 100.0,
        (base - float(prefix["TVT"].iloc[-1])) / 100.0,
        base_gradient * 1000.0,
        np.full(expected, u_slope * 1000.0),
        np.full(expected, float(requested_fraction)),
        np.full(expected, np.log1p(len(prefix))),
    ]
    names = [
        "suffix_progress",
        "md_from_cut_kft",
        "gr_prefix_z",
        "gr_last_z",
        "z_from_cut_kft",
        "base_move_hft",
        "base_from_last_tvt_hft",
        "base_gradient_per_kft",
        "prefix_u_slope_per_kft",
        "requested_fraction",
        "log_prefix_rows",
    ]
    deltas = []
    for name in sorted(candidates):
        if name == base_name:
            continue
        delta = np.asarray(candidates[name], float) - base
        values.append(delta / 10.0)
        names.append(f"delta_{name}_10ft")
        deltas.append(delta)
    disagreement = np.stack(deltas, axis=1)
    values.extend(
        [
            np.mean(disagreement, axis=1) / 10.0,
            np.std(disagreement, axis=1) / 10.0,
            np.max(np.abs(disagreement), axis=1) / 10.0,
        ]
    )
    names.extend(["delta_mean_10ft", "delta_std_10ft", "delta_absmax_10ft"])
    matrix = np.column_stack(values).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Residual feature matrix contains non-finite values")
    return matrix, names


def _correction_prediction(
    base: np.ndarray,
    raw_residual: np.ndarray,
    weight: float,
    cap_ft: float,
    ramp_rows: float,
) -> np.ndarray:
    step = np.arange(1, len(base) + 1, dtype=float)
    ramp = 1.0 - np.exp(-step / max(float(ramp_rows), 1.0))
    move = np.clip(float(weight) * ramp * np.asarray(raw_residual, float), -cap_ft, cap_ft)
    return np.asarray(base, float) + move


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    stage16 = args.stage16b_run.resolve()
    stage17 = args.stage17a_run.resolve()
    public_run = args.public_oof_run.resolve()
    training_run = args.training_run.resolve()
    validation_run = args.validation_run.resolve()
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    for run, label in [(stage17, "Stage 17A"), (training_run, "Stage 21A"), (validation_run, "Stage 21B")]:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        if summary.get("stage16b_manifest_sha256") != expected_hash:
            raise AssertionError(f"{label} manifest provenance mismatch")
    if json.loads((validation_run / "summary.json").read_text()).get("promoted_to_stage21c") is not False:
        raise AssertionError("Stage 22A expects the rejected Stage 21B validation split")

    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    all_cuts = pd.read_parquet(stage17 / "cut_report.parquet")
    training_ids = pd.read_parquet(training_run / "router_cut_report.parquet", columns=["cut_id"])
    validation_ids = pd.read_parquet(validation_run / "confidence_cut_report.parquet", columns=["cut_id"])
    training = all_cuts[all_cuts["cut_id"].isin(training_ids["cut_id"])].copy()
    validation = all_cuts[all_cuts["cut_id"].isin(validation_ids["cut_id"])].copy()
    if args.limit_training_cuts is not None:
        training = training.sort_values("cut_id").head(int(args.limit_training_cuts)).copy()
    if args.limit_validation_cuts is not None:
        validation = validation.sort_values("cut_id").head(int(args.limit_validation_cuts)).copy()
    training_wells = set(training["well_id"].astype(str))
    validation_wells = set(validation["well_id"].astype(str))
    overlap = sorted(training_wells.intersection(validation_wells))
    if overlap:
        raise AssertionError(f"Training/validation well leakage: {overlap[:5]}")

    assignments = pd.read_parquet(stage16 / "well_assignments.parquet")
    assignments["well_id"] = assignments["well_id"].astype(str)
    assignments["typewell_fold"] = _typewell_folds(assignments, 5, seed)
    validation = validation.merge(
        assignments[["well_id", "spatial_fold", "typewell_fold", "branch_group_fold"]],
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    validation["stage16_fold"] = validation["stage16_fold"].astype(int)
    wells = sorted(training_wells.union(validation_wells))
    public = pd.read_parquet(
        public_run / "base_oof.parquet",
        columns=["well_id", "row_index", "y_pred"],
        filters=[("well_id", "in", wells)],
    )
    public["well_id"] = public["well_id"].astype(str)
    public_by_well = {
        well: frame.sort_values("row_index") for well, frame in public.groupby("well_id", sort=True)
    }
    train_dir = args.data_dir.resolve() / "train"

    @lru_cache(maxsize=64)
    def load_well(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train_dir / f"{well_id}__horizontal_well.csv")

    @lru_cache(maxsize=64)
    def load_typewell(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train_dir / f"{well_id}__typewell.csv")

    candidate_config = dict(config.get("candidates", {}))
    model_config = dict(config.get("model", {}))
    base_name = str(model_config.get("base_candidate", "top_pf_a130"))
    stride = int(model_config.get("training_stride", 4))
    target_cap = float(model_config.get("target_cap_ft", 24.0))
    train_x, train_y, train_weight = [], [], []
    feature_names: list[str] | None = None
    for position, cut in enumerate(training.sort_values("cut_id").itertuples(index=False), 1):
        well_id, outer = str(cut.well_id), int(cut.cut_index)
        horizontal, typewell = load_well(well_id), load_typewell(well_id)
        source = public_by_well[well_id]
        original = int(source["row_index"].min())
        candidates = build_candidates(
            horizontal,
            typewell,
            outer,
            source["y_pred"].to_numpy(float)[outer - original :],
            candidate_config,
        )
        features, names = residual_features(
            horizontal, outer, candidates, base_name, float(cut.requested_fraction)
        )
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise AssertionError("Feature order changed")
        index = np.arange(0, len(features), stride, dtype=int)
        truth = horizontal["TVT"].to_numpy(float)[outer:]
        residual = np.clip(truth - candidates[base_name], -target_cap, target_cap)
        train_x.append(features[index])
        train_y.append(residual[index].astype(np.float32))
        train_weight.append(np.full(len(index), 1.0 / max(len(index), 1), dtype=np.float32))
        if position % 10 == 0:
            print(f"residual field training features {position}/{len(training)} cuts", flush=True)
    x = np.concatenate(train_x)
    y = np.concatenate(train_y)
    sample_weight = np.concatenate(train_weight)
    sample_weight *= len(sample_weight) / sample_weight.sum()
    model = HistGradientBoostingRegressor(
        learning_rate=float(model_config.get("learning_rate", 0.05)),
        max_iter=int(model_config.get("max_iter", 180)),
        max_leaf_nodes=int(model_config.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(model_config.get("min_samples_leaf", 80)),
        l2_regularization=float(model_config.get("l2_regularization", 2.0)),
        random_state=seed,
    )
    model.fit(x, y, sample_weight=sample_weight)

    correction = dict(config.get("correction", {}))
    weights = [float(value) for value in correction.get("diagnostic_weights", [0.25])]
    primary_weight = float(correction.get("primary_weight", 0.25))
    if primary_weight not in weights:
        weights.append(primary_weight)
    cut_rows: list[dict[str, Any]] = []
    invariance: list[bool] = []
    for position, cut in enumerate(validation.sort_values("cut_id").itertuples(index=False), 1):
        well_id, outer = str(cut.well_id), int(cut.cut_index)
        horizontal, typewell = load_well(well_id), load_typewell(well_id)
        source = public_by_well[well_id]
        original = int(source["row_index"].min())
        candidates = build_candidates(
            horizontal,
            typewell,
            outer,
            source["y_pred"].to_numpy(float)[outer - original :],
            candidate_config,
        )
        features, names = residual_features(
            horizontal, outer, candidates, base_name, float(cut.requested_fraction)
        )
        if names != feature_names:
            raise AssertionError("Validation feature order changed")
        raw_residual = model.predict(features)
        base = candidates[base_name]
        truth = horizontal["TVT"].to_numpy(float)[outer:]
        row: dict[str, Any] = {
            "cut_id": str(cut.cut_id),
            "well_id": well_id,
            "requested_fraction": float(cut.requested_fraction),
            **{family: int(getattr(cut, family)) for family in FOLD_FAMILIES},
            "suffix_rows": len(truth),
            "base_sse": float(np.square(base - truth).sum()),
            "raw_residual_mean": float(np.mean(raw_residual)),
            "raw_residual_std": float(np.std(raw_residual)),
        }
        for weight in weights:
            prediction = _correction_prediction(
                base,
                raw_residual,
                weight,
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
                changed,
                typewell,
                outer,
                source["y_pred"].to_numpy(float)[outer - original :],
                candidate_config,
            )
            changed_features, _ = residual_features(
                changed, outer, changed_candidates, base_name, float(cut.requested_fraction)
            )
            invariance.append(np.array_equal(features, changed_features))
        if position % 10 == 0:
            print(f"residual field validation {position}/{len(validation)} cuts", flush=True)

    report = pd.DataFrame.from_records(cut_rows)
    primary_column = f"candidate_sse_w{int(round(primary_weight * 1000)):03d}"
    report["candidate_sse"] = report[primary_column]
    metrics = _metrics(report)
    bootstrap = _bootstrap(
        report, int(config.get("validation", {}).get("bootstrap_resamples", 3000)), seed
    )
    family_reports = {family: _family_report(report, family) for family in FOLD_FAMILIES}
    base_well = report.groupby("well_id").agg(sse=("base_sse", "sum"), rows=("suffix_rows", "sum"))
    candidate_well = report.groupby("well_id").agg(
        sse=("candidate_sse", "sum"), rows=("suffix_rows", "sum")
    )
    p90_delta = float(
        np.sqrt(candidate_well.sse / candidate_well.rows).quantile(0.9)
        - np.sqrt(base_well.sse / base_well.rows).quantile(0.9)
    )
    rows = int(report["suffix_rows"].sum())
    base_rmse = float(np.sqrt(report["base_sse"].sum() / rows))
    weight_report = []
    for weight in sorted(weights):
        column = f"candidate_sse_w{int(round(weight * 1000)):03d}"
        candidate_rmse = float(np.sqrt(report[column].sum() / rows))
        weight_report.append(
            {
                "weight": weight,
                "base_rmse": base_rmse,
                "candidate_rmse": candidate_rmse,
                "rmse_delta": candidate_rmse - base_rmse,
            }
        )
    validation_config = dict(config.get("validation", {}))
    gates = {
        "hidden_target_invariance": bool(invariance) and all(invariance),
        "training_validation_well_overlap_zero": len(overlap) == 0,
        "primary_gain": metrics["rmse_delta"]
        <= -float(validation_config.get("minimum_gain", 0.1)),
        "bootstrap_upper_below_zero": bootstrap[1] < 0,
        "standard_fold_consistency": metrics["improved_folds"]
        >= int(validation_config.get("minimum_improved_folds", 4)),
        "fraction_consistency": metrics["improved_fractions"]
        >= int(validation_config.get("minimum_improved_fractions", 4)),
        "spatial_fold_consistency": family_reports["spatial_fold"]["improved_folds"]
        >= int(validation_config.get("minimum_spatial_folds", 4)),
        "typewell_fold_consistency": family_reports["typewell_fold"]["improved_folds"]
        >= int(validation_config.get("minimum_typewell_folds", 4)),
        "branch_group_fold_consistency": family_reports["branch_group_fold"]["improved_folds"]
        >= int(validation_config.get("minimum_branch_folds", 4)),
        "well_p90_nonworse": p90_delta <= 0,
    }
    promoted = bool(all(gates.values()))
    report.to_parquet(output / "residual_field_cut_report.parquet", index=False)
    summary = {
        "stage22a_complete": True,
        "promoted_to_stage22b": promoted,
        "stage16b_manifest_sha256": expected_hash,
        "training_cuts": len(training),
        "training_wells": len(training_wells),
        "training_rows": int(len(x)),
        "validation_cuts": len(validation),
        "validation_wells": len(validation_wells),
        "training_validation_well_overlap": overlap,
        "feature_count": len(feature_names or []),
        "base_rmse": metrics["base_rmse"],
        "candidate_rmse": metrics["candidate_rmse"],
        "rmse_delta": metrics["rmse_delta"],
        "well_p90_delta": p90_delta,
        "bootstrap_95pct": bootstrap,
        "metrics": metrics,
        "family_reports": family_reports,
        "weight_report": weight_report,
        "gates": gates,
        "next_step": (
            "Run a second disjoint confirmation with fold-safe training expansion."
            if promoted
            else "Reject candidate-disagreement residual fields and redesign the physical alignment state."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    write_json(output / "feature_names.json", feature_names)
    config["resolved"] = {
        "stage16b_run": str(stage16),
        "stage17a_run": str(stage17),
        "public_oof_run": str(public_run),
        "training_run": str(training_run),
        "validation_run": str(validation_run),
        "data_dir": str(args.data_dir.resolve()),
        "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
