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
from rogii.cli.expanded_residual_field import expanded_residual_features, smooth_residual_target
from rogii.cli.prefix_residual_field import _correction_prediction
from rogii.cli.prefix_router import FOLD_FAMILIES, _family_report, build_candidates
from rogii.cli.trajectory_residual import _typewell_folds
from rogii.config import load_config
from rogii.models.functional_residual import (
    curve_descriptor,
    equal_cut_optimal_alpha,
    resample_curve,
    restore_curve,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 35A functional PCA residual curve")
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


def _coefficient_model(config: dict[str, Any], seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=float(config.get("learning_rate", 0.04)),
        max_iter=int(config.get("max_iter", 240)),
        max_leaf_nodes=int(config.get("max_leaf_nodes", 15)),
        min_samples_leaf=int(config.get("min_samples_leaf", 30)),
        l2_regularization=float(config.get("l2_regularization", 8.0)),
        loss="squared_error",
        early_stopping=False,
        random_state=int(seed),
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    stage16 = args.stage16b_run.resolve()
    stage17 = args.stage17a_run.resolve()
    public_run = args.public_oof_run.resolve()
    manifest_run = args.split_manifest_run.resolve()
    validation_run = args.validation_run.resolve()
    manifest = json.loads((manifest_run / "summary.json").read_text(encoding="utf-8"))
    summary17 = json.loads((stage17 / "summary.json").read_text(encoding="utf-8"))
    if summary17.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 17A provenance mismatch")
    if manifest.get("training_wells") != 500 or manifest.get("confirmation_wells") != 120:
        raise AssertionError("Stage 35A requires the frozen 500/120 split")
    if manifest.get("reserved_confirmation_used") is not False:
        raise AssertionError("Split manifest opened the confirmation reserve")
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
    if training_wells & validation_wells:
        raise AssertionError("Training/design-validation well overlap")
    assignments = pd.read_parquet(stage16 / "well_assignments.parquet")
    assignments["well_id"] = assignments["well_id"].astype(str)
    assignments["typewell_fold"] = _typewell_folds(assignments, 5, seed)
    assignment_columns = ["well_id", "spatial_fold", "typewell_fold", "branch_group_fold"]
    training = training.merge(assignments[assignment_columns], on="well_id", how="left", validate="many_to_one")
    validation = validation.merge(assignments[assignment_columns], on="well_id", how="left", validate="many_to_one")
    for frame in (training, validation):
        frame["stage16_fold"] = frame["stage16_fold"].astype(int)
    wells = sorted(training_wells | validation_wells)
    public = pd.read_parquet(
        public_run / "base_oof.parquet",
        columns=["well_id", "row_index", "y_pred"],
        filters=[("well_id", "in", wells)],
    )
    public["well_id"] = public["well_id"].astype(str)
    public_by_well = {
        well: frame.sort_values("row_index") for well, frame in public.groupby("well_id")
    }
    train_dir = args.data_dir.resolve() / "train"

    @lru_cache(maxsize=64)
    def load_well(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train_dir / f"{well_id}__horizontal_well.csv")

    @lru_cache(maxsize=64)
    def load_typewell(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train_dir / f"{well_id}__typewell.csv")

    candidate_config = dict(config.get("candidates", {}))
    feature_config = dict(config.get("features", {}))
    functional_config = dict(config.get("functional_target", {}))
    correction = dict(config.get("correction", {}))
    base_name = str(config.get("base_candidate", "top_pf_a130"))
    grid_points = int(functional_config.get("grid_points", 96))
    component_count = int(functional_config.get("components", 8))
    smoothing_rows = int(functional_config.get("smoothing_rows", 101))
    target_cap = float(functional_config.get("target_cap_ft", 24.0))
    correction_cap = float(correction.get("cap_ft", 8.0))
    ramp_rows = float(correction.get("ramp_rows", 96.0))
    maximum_alpha = float(correction.get("maximum_oof_alpha", 0.75))
    descriptor_names: list[str] | None = None

    def build_cut(cut: Any) -> dict[str, Any]:
        well_id, outer = str(cut.well_id), int(cut.cut_index)
        horizontal = load_well(well_id)
        typewell = load_typewell(well_id)
        source = public_by_well[well_id]
        original = int(source["row_index"].min())
        candidates = build_candidates(
            horizontal,
            typewell,
            outer,
            source["y_pred"].to_numpy(float)[outer - original :],
            candidate_config,
        )
        row_features, _ = expanded_residual_features(
            horizontal,
            outer,
            candidates,
            base_name,
            float(cut.requested_fraction),
            plane_window_ft=float(feature_config.get("plane_window_ft", 1200.0)),
            plane_ridge=float(feature_config.get("plane_ridge", 0.05)),
        )
        descriptor, names = curve_descriptor(
            row_features, candidates, base_name, float(cut.requested_fraction)
        )
        base = np.asarray(candidates[base_name], float)
        truth = horizontal["TVT"].to_numpy(float)[outer:]
        return {
            "descriptor": descriptor,
            "descriptor_names": names,
            "base": base,
            "truth": truth,
            "target_grid": resample_curve(
                smooth_residual_target(truth - base, smoothing_rows, target_cap),
                grid_points,
            ),
            "row_features": row_features,
            "candidates": candidates,
        }

    training_items = []
    training_metadata = []
    for position, cut in enumerate(training.itertuples(index=False), 1):
        item = build_cut(cut)
        descriptor_names = (
            item["descriptor_names"] if descriptor_names is None else descriptor_names
        )
        if item["descriptor_names"] != descriptor_names:
            raise AssertionError("Functional descriptor schema changed")
        training_items.append(item)
        training_metadata.append(
            {
                "cut_id": str(cut.cut_id),
                "well_id": str(cut.well_id),
                "requested_fraction": float(cut.requested_fraction),
                **{family: int(getattr(cut, family)) for family in FOLD_FAMILIES},
                "suffix_rows": len(item["truth"]),
            }
        )
        if position % 25 == 0:
            print(f"FPCA training curves {position}/{len(training)} cuts", flush=True)
    descriptors = np.stack([item["descriptor"] for item in training_items])
    curves = np.stack([item["target_grid"] for item in training_items]).astype(float)
    metadata = pd.DataFrame(training_metadata)
    model_config = dict(config.get("coefficient_model", {}))
    fold_packages: dict[int, dict[str, Any]] = {}
    oof_grids = np.zeros_like(curves)
    explained_variances = []
    for fold in range(5):
        train_mask = metadata["stage16_fold"].to_numpy(int) != fold
        valid_mask = ~train_mask
        fold_curves = curves[train_mask]
        fold_mean = fold_curves.mean(axis=0)
        centered = fold_curves - fold_mean
        _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
        fold_components = right[:component_count]
        fold_coefficients = centered @ fold_components.T
        total_variance = float(np.square(singular_values).sum())
        explained_variances.append(
            float(
                np.square(singular_values[:component_count]).sum()
                / max(total_variance, 1e-12)
            )
        )
        fold_models = []
        predicted_coefficients = np.zeros((int(valid_mask.sum()), component_count), float)
        for component in range(component_count):
            model = _coefficient_model(model_config, seed + 100 * fold + component)
            model.fit(descriptors[train_mask], fold_coefficients[:, component])
            predicted_coefficients[:, component] = model.predict(descriptors[valid_mask])
            fold_models.append(model)
        oof_grids[valid_mask] = fold_mean[None] + predicted_coefficients @ fold_components
        fold_packages[fold] = {
            "mean": fold_mean,
            "components": fold_components,
            "models": fold_models,
        }
        print(f"trained FPCA coefficient fold {fold}", flush=True)
    explained_variance = float(np.mean(explained_variances))
    oof_full = []
    bases = []
    truths = []
    for item, grid in zip(training_items, oof_grids, strict=True):
        residual = restore_curve(grid, len(item["truth"]))
        full = _correction_prediction(
            item["base"], residual, 1.0, correction_cap, ramp_rows
        )
        bases.append(item["base"])
        truths.append(item["truth"])
        oof_full.append(full)
    alpha = equal_cut_optimal_alpha(bases, oof_full, truths, maximum_alpha)
    oof_rows = []
    for meta, base, full, truth in zip(
        training_metadata, bases, oof_full, truths, strict=True
    ):
        candidate = base + alpha * (full - base)
        oof_rows.append(
            {
                **meta,
                "base_sse": float(np.square(base - truth).sum()),
                "candidate_sse": float(np.square(candidate - truth).sum()),
            }
        )
    oof_report = pd.DataFrame(oof_rows)
    oof_metrics = _metrics(oof_report)
    oof_bootstrap = _bootstrap(
        oof_report, int(config.get("validation", {}).get("bootstrap_resamples", 4000)), seed
    )

    validation_rows = []
    invariance = []
    for position, cut in enumerate(validation.itertuples(index=False), 1):
        item = build_cut(cut)
        if item["descriptor_names"] != descriptor_names:
            raise AssertionError("Validation functional descriptor schema changed")
        fold_grids = []
        for fold in range(5):
            package = fold_packages[fold]
            predicted_coefficients = np.asarray(
                [
                    package["models"][component].predict(item["descriptor"][None])[0]
                    for component in range(component_count)
                ],
                float,
            )
            fold_grids.append(
                package["mean"] + predicted_coefficients @ package["components"]
            )
        grid = np.mean(fold_grids, axis=0)
        residual = restore_curve(grid, len(item["truth"]))
        full = _correction_prediction(
            item["base"], residual, 1.0, correction_cap, ramp_rows
        )
        candidate = item["base"] + alpha * (full - item["base"])
        validation_rows.append(
            {
                "cut_id": str(cut.cut_id),
                "well_id": str(cut.well_id),
                "requested_fraction": float(cut.requested_fraction),
                **{family: int(getattr(cut, family)) for family in FOLD_FAMILIES},
                "suffix_rows": len(item["truth"]),
                "base_sse": float(np.square(item["base"] - item["truth"]).sum()),
                "candidate_sse": float(np.square(candidate - item["truth"]).sum()),
                "predicted_residual_mean": float(residual.mean()),
                "predicted_residual_std": float(residual.std()),
            }
        )
        if len(invariance) < 8:
            changed = load_well(str(cut.well_id)).copy()
            changed.loc[changed.index >= int(cut.cut_index), "TVT"] += 9999.0
            source = public_by_well[str(cut.well_id)]
            original = int(source["row_index"].min())
            changed_candidates = build_candidates(
                changed,
                load_typewell(str(cut.well_id)),
                int(cut.cut_index),
                source["y_pred"].to_numpy(float)[int(cut.cut_index) - original :],
                candidate_config,
            )
            changed_features, _ = expanded_residual_features(
                changed,
                int(cut.cut_index),
                changed_candidates,
                base_name,
                float(cut.requested_fraction),
                plane_window_ft=float(feature_config.get("plane_window_ft", 1200.0)),
                plane_ridge=float(feature_config.get("plane_ridge", 0.05)),
            )
            changed_descriptor, changed_names = curve_descriptor(
                changed_features,
                changed_candidates,
                base_name,
                float(cut.requested_fraction),
            )
            invariance.append(
                changed_names == descriptor_names
                and np.array_equal(item["descriptor"], changed_descriptor)
            )
        if position % 10 == 0:
            print(f"FPCA validation {position}/{len(validation)} cuts", flush=True)

    report = pd.DataFrame(validation_rows)
    metrics = _metrics(report)
    bootstrap = _bootstrap(
        report, int(config.get("validation", {}).get("bootstrap_resamples", 4000)), seed
    )
    family_reports = {family: _family_report(report, family) for family in FOLD_FAMILIES}
    validation_config = dict(config.get("validation", {}))
    gates = {
        "hidden_target_invariance": bool(invariance) and all(invariance),
        "training_validation_well_overlap_zero": not bool(training_wells & validation_wells),
        "oof_alpha_positive": alpha > 0.0,
        "oof_gain": oof_metrics["rmse_delta"] < 0.0,
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
    np.savez(
        output / "fpca_basis.npz",
        means=np.stack([fold_packages[fold]["mean"] for fold in range(5)]).astype(np.float32),
        components=np.stack(
            [fold_packages[fold]["components"] for fold in range(5)]
        ).astype(np.float32),
    )
    oof_report.to_parquet(output / "training_oof_report.parquet", index=False)
    report.to_parquet(output / "validation_report.parquet", index=False)
    summary = {
        "stage35a_complete": True,
        "promoted_to_stage35b_reserved_confirmation": promoted,
        "stage16b_manifest_sha256": expected_hash,
        "training_cuts": len(training_items),
        "training_wells": len(training_wells),
        "validation_cuts": len(report),
        "validation_wells": len(validation_wells),
        "descriptor_features": len(descriptor_names or []),
        "grid_points": grid_points,
        "components": component_count,
        "explained_variance": explained_variance,
        "fold_safe_basis": True,
        "oof_alpha": alpha,
        "oof_rmse_delta": oof_metrics["rmse_delta"],
        "oof_bootstrap_95pct": oof_bootstrap,
        "base_rmse": metrics["base_rmse"],
        "candidate_rmse": metrics["candidate_rmse"],
        "rmse_delta": metrics["rmse_delta"],
        "bootstrap_95pct": bootstrap,
        "cut_rmse_p90_delta": metrics["cut_rmse_p90_delta"],
        "family_reports": family_reports,
        "gates": gates,
        "reserved_confirmation_used": False,
        "next_step": (
            "Freeze the FPCA basis/models and run exactly one Stage 35B reserved confirmation."
            if promoted
            else "Reject the functional residual curve without opening the confirmation reserve."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16),
        "stage17a_run": str(stage17),
        "public_oof_run": str(public_run),
        "split_manifest_run": str(manifest_run),
        "validation_run": str(validation_run),
        "data_dir": str(args.data_dir.resolve()),
        "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
