from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.branch_retrieval import _bootstrap, _metrics
from rogii.cli.expanded_residual_field import expanded_residual_features
from rogii.cli.prefix_residual_field import _correction_prediction
from rogii.cli.prefix_router import FOLD_FAMILIES, _family_report, build_candidates
from rogii.cli.sequence_residual_field import _device, _make_model, _predict
from rogii.cli.trajectory_residual import _typewell_folds
from rogii.config import load_config
from rogii.models.residual_cut_gate import cut_gate_features, optimal_gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 33A fold-safe TCN cut-benefit gate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--public-oof-run", type=Path, required=True)
    parser.add_argument("--stage30a-run", type=Path, required=True)
    parser.add_argument("--split-manifest-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit-training-cuts", type=int)
    return parser


def _gate_model(config: dict[str, Any], seed: int) -> HistGradientBoostingRegressor:
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


def _load_tcn_models(
    stage30: Path, device: torch.device
) -> tuple[dict[int, torch.nn.Module], list[str], np.ndarray, np.ndarray, float]:
    normalizer = np.load(stage30 / "normalizer.npz")
    feature_mean = normalizer["mean"].astype(np.float32)
    feature_scale = normalizer["scale"].astype(np.float32)
    models: dict[int, torch.nn.Module] = {}
    feature_names: list[str] | None = None
    target_scale: float | None = None
    for path in sorted(stage30.glob("fold_*.pt")):
        fold = int(path.stem.split("_")[-1])
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = _make_model(int(checkpoint["feature_count"]), checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        models[fold] = model.to(device).eval()
        names = list(checkpoint["feature_names"])
        feature_names = names if feature_names is None else feature_names
        if names != feature_names:
            raise AssertionError("Stage 30 checkpoint feature schemas differ")
        scale = float(checkpoint["model_config"].get("target_scale_ft", 10.0))
        target_scale = scale if target_scale is None else target_scale
        if scale != target_scale:
            raise AssertionError("Stage 30 checkpoint target scales differ")
    if sorted(models) != [0, 1, 2, 3, 4] or feature_names is None or target_scale is None:
        raise AssertionError(f"Expected Stage 30 folds 0..4, found {sorted(models)}")
    return models, feature_names, feature_mean, feature_scale, target_scale


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    stage16 = args.stage16b_run.resolve()
    stage17 = args.stage17a_run.resolve()
    public_run = args.public_oof_run.resolve()
    stage30 = args.stage30a_run.resolve()
    manifest_run = args.split_manifest_run.resolve()
    validation_run = args.validation_run.resolve()
    summary30 = json.loads((stage30 / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((manifest_run / "summary.json").read_text(encoding="utf-8"))
    if summary30.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 30A provenance mismatch")
    if not summary30.get("stage30a_complete") or summary30.get("reserved_confirmation_used") is not False:
        raise AssertionError("Stage 30A is incomplete or opened the reserve")
    if manifest.get("training_wells") != 500 or manifest.get("confirmation_wells") != 120:
        raise AssertionError("Stage 33A requires the frozen 500/120 split")
    if manifest.get("reserved_confirmation_used") is not False:
        raise AssertionError("Split manifest opened the confirmation reserve")
    summary17 = json.loads((stage17 / "summary.json").read_text(encoding="utf-8"))
    if summary17.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 17A provenance mismatch")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    device = _device(args.device)
    models, tcn_feature_names, feature_mean, feature_scale, target_scale = _load_tcn_models(
        stage30, device
    )
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
    correction = dict(config.get("correction", {}))
    base_name = str(config.get("base_candidate", "top_pf_a130"))
    weight = float(correction.get("weight", 0.30))
    cap = float(correction.get("cap_ft", 8.0))
    ramp = float(correction.get("ramp_rows", 96.0))
    gate_feature_names: list[str] | None = None

    def build_inputs(cut: Any) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
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
        features, names = expanded_residual_features(
            horizontal,
            outer,
            candidates,
            base_name,
            float(cut.requested_fraction),
            plane_window_ft=float(feature_config.get("plane_window_ft", 1200.0)),
            plane_ridge=float(feature_config.get("plane_ridge", 0.05)),
        )
        if names != tcn_feature_names:
            raise AssertionError("Stage 33 feature schema differs from Stage 30")
        normalized = ((features - feature_mean) / feature_scale).astype(np.float32)
        truth = horizontal["TVT"].to_numpy(float)[outer:]
        return normalized, candidates, truth, features

    training_rows: list[dict[str, Any]] = []
    training_vectors = []
    training_targets = []
    for position, cut in enumerate(training.itertuples(index=False), 1):
        normalized, candidates, truth, _ = build_inputs(cut)
        fold = int(cut.stage16_fold)
        raw = target_scale * _predict(models[fold], normalized, device)
        vector, names = cut_gate_features(
            normalized, raw, candidates, base_name, float(cut.requested_fraction)
        )
        gate_feature_names = names if gate_feature_names is None else gate_feature_names
        if names != gate_feature_names:
            raise AssertionError("Cut gate feature schema changed")
        base = candidates[base_name]
        full = _correction_prediction(base, raw, weight, cap, ramp)
        target_gate = optimal_gate(base, full, truth)
        training_vectors.append(vector)
        training_targets.append(target_gate)
        training_rows.append(
            {
                "cut_id": str(cut.cut_id),
                "well_id": str(cut.well_id),
                "stage16_fold": fold,
                "requested_fraction": float(cut.requested_fraction),
                "suffix_rows": len(truth),
                "base_sse": float(np.square(base - truth).sum()),
                "full_sse": float(np.square(full - truth).sum()),
                "optimal_gate": target_gate,
            }
        )
        if position % 25 == 0:
            print(f"cut benefit training features {position}/{len(training)} cuts", flush=True)
    x_train = np.stack(training_vectors)
    y_train = np.asarray(training_targets, float)
    training_report = pd.DataFrame(training_rows)
    model_config = dict(config.get("gate_model", {}))
    gate_models = []
    crossfit_gate = np.zeros(len(training_report), float)
    for fold in range(5):
        train_mask = training_report["stage16_fold"].to_numpy(int) != fold
        valid_mask = ~train_mask
        model = _gate_model(model_config, seed + fold)
        model.fit(x_train[train_mask], y_train[train_mask])
        crossfit_gate[valid_mask] = np.clip(model.predict(x_train[valid_mask]), 0.0, 1.0)
        gate_models.append(model)
    training_report["crossfit_gate"] = crossfit_gate
    training_report["crossfit_gate_mae"] = np.abs(crossfit_gate - y_train)

    rows = []
    invariance = []
    for position, cut in enumerate(validation.itertuples(index=False), 1):
        normalized, candidates, truth, original_features = build_inputs(cut)
        predictions = np.stack(
            [target_scale * _predict(models[fold], normalized, device) for fold in range(5)]
        )
        raw = predictions.mean(axis=0)
        vector, names = cut_gate_features(
            normalized, raw, candidates, base_name, float(cut.requested_fraction)
        )
        if names != gate_feature_names:
            raise AssertionError("Validation cut gate feature schema changed")
        predicted_gate = float(
            np.clip(np.mean([model.predict(vector[None])[0] for model in gate_models]), 0.0, 1.0)
        )
        base = candidates[base_name]
        full = _correction_prediction(base, raw, weight, cap, ramp)
        candidate = base + predicted_gate * (full - base)
        row: dict[str, Any] = {
            "cut_id": str(cut.cut_id),
            "well_id": str(cut.well_id),
            "requested_fraction": float(cut.requested_fraction),
            **{family: int(getattr(cut, family)) for family in FOLD_FAMILIES},
            "suffix_rows": len(truth),
            "base_sse": float(np.square(base - truth).sum()),
            "full_sse": float(np.square(full - truth).sum()),
            "candidate_sse": float(np.square(candidate - truth).sum()),
            "predicted_gate": predicted_gate,
            "oracle_gate": optimal_gate(base, full, truth),
            "ensemble_spread_mean": float(predictions.std(axis=0).mean()),
        }
        rows.append(row)
        if len(invariance) < 8:
            horizontal = load_well(str(cut.well_id)).copy()
            horizontal.loc[horizontal.index >= int(cut.cut_index), "TVT"] += 9999.0
            source = public_by_well[str(cut.well_id)]
            original = int(source["row_index"].min())
            changed_candidates = build_candidates(
                horizontal,
                load_typewell(str(cut.well_id)),
                int(cut.cut_index),
                source["y_pred"].to_numpy(float)[int(cut.cut_index) - original :],
                candidate_config,
            )
            changed_features, _ = expanded_residual_features(
                horizontal,
                int(cut.cut_index),
                changed_candidates,
                base_name,
                float(cut.requested_fraction),
                plane_window_ft=float(feature_config.get("plane_window_ft", 1200.0)),
                plane_ridge=float(feature_config.get("plane_ridge", 0.05)),
            )
            invariance.append(np.array_equal(original_features, changed_features))
        if position % 10 == 0:
            print(f"cut benefit validation {position}/{len(validation)} cuts", flush=True)

    report = pd.DataFrame(rows)
    metrics = _metrics(report)
    bootstrap = _bootstrap(
        report, int(config.get("validation", {}).get("bootstrap_resamples", 4000)), seed
    )
    family_reports = {family: _family_report(report, family) for family in FOLD_FAMILIES}
    validation_config = dict(config.get("validation", {}))
    gates = {
        "hidden_target_invariance": bool(invariance) and all(invariance),
        "training_validation_well_overlap_zero": not bool(training_wells & validation_wells),
        "fold_safe_tcn_training_predictions": True,
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
    training_report.to_parquet(output / "training_gate_report.parquet", index=False)
    report.to_parquet(output / "validation_gate_report.parquet", index=False)
    summary = {
        "stage33a_complete": True,
        "promoted_to_stage33b_reserved_confirmation": promoted,
        "stage16b_manifest_sha256": expected_hash,
        "device": str(device),
        "training_cuts": len(training_report),
        "training_wells": len(training_wells),
        "validation_cuts": len(report),
        "validation_wells": len(validation_wells),
        "gate_feature_count": len(gate_feature_names or []),
        "gate_models": len(gate_models),
        "training_optimal_gate_mean": float(y_train.mean()),
        "training_crossfit_gate_mae": float(np.mean(np.abs(crossfit_gate - y_train))),
        "validation_predicted_gate_mean": float(report["predicted_gate"].mean()),
        "validation_oracle_gate_mean": float(report["oracle_gate"].mean()),
        "base_rmse": metrics["base_rmse"],
        "candidate_rmse": metrics["candidate_rmse"],
        "rmse_delta": metrics["rmse_delta"],
        "bootstrap_95pct": bootstrap,
        "cut_rmse_p90_delta": metrics["cut_rmse_p90_delta"],
        "family_reports": family_reports,
        "gates": gates,
        "reserved_confirmation_used": False,
        "next_step": (
            "Freeze the learned cut gate and run exactly one Stage 33B reserved confirmation."
            if promoted
            else "Reject the learned cut-benefit gate without opening the confirmation reserve."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16),
        "stage17a_run": str(stage17),
        "public_oof_run": str(public_run),
        "stage30a_run": str(stage30),
        "split_manifest_run": str(manifest_run),
        "validation_run": str(validation_run),
        "data_dir": str(args.data_dir.resolve()),
        "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
