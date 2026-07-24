from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.branch_retrieval import _bootstrap, _metrics
from rogii.cli.expanded_residual_field import expanded_residual_features
from rogii.cli.prefix_residual_field import _correction_prediction
from rogii.cli.prefix_router import FOLD_FAMILIES, _family_report, build_candidates
from rogii.cli.sequence_residual_field import _device, _make_model, _predict
from rogii.cli.trajectory_residual import _typewell_folds
from rogii.config import load_config
from rogii.models.residual_uncertainty import uncertainty_shrunk_residual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 31A target-free TCN uncertainty shrinkage")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--public-oof-run", type=Path, required=True)
    parser.add_argument("--stage30a-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    stage16, stage17 = args.stage16b_run.resolve(), args.stage17a_run.resolve()
    public_run, stage30 = args.public_oof_run.resolve(), args.stage30a_run.resolve()
    validation_run = args.validation_run.resolve()
    summary30 = json.loads((stage30 / "summary.json").read_text(encoding="utf-8"))
    if summary30.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 30A provenance mismatch")
    if not summary30.get("stage30a_complete"):
        raise AssertionError("Stage 30A did not complete")
    if summary30.get("reserved_confirmation_used") is not False:
        raise AssertionError("Stage 30A opened the confirmation reserve")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    normalizer = np.load(stage30 / "normalizer.npz")
    feature_mean = normalizer["mean"].astype(np.float32)
    feature_scale = normalizer["scale"].astype(np.float32)
    models = []
    feature_names = None
    for path in sorted(stage30.glob("fold_*.pt")):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = _make_model(int(checkpoint["feature_count"]), checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        models.append(model.to(device).eval())
        names = list(checkpoint["feature_names"])
        feature_names = names if feature_names is None else feature_names
        if names != feature_names:
            raise AssertionError("Stage 30 checkpoint feature schemas differ")
    if len(models) != 5:
        raise AssertionError(f"Expected five Stage 30 models, found {len(models)}")

    cuts = pd.read_parquet(stage17 / "cut_report.parquet")
    validation_ids = pd.read_parquet(
        validation_run / "confidence_cut_report.parquet", columns=["cut_id"]
    )
    validation = cuts[cuts["cut_id"].isin(validation_ids["cut_id"])].copy().sort_values("cut_id")
    assignments = pd.read_parquet(stage16 / "well_assignments.parquet")
    assignments["well_id"] = assignments["well_id"].astype(str)
    assignments["typewell_fold"] = _typewell_folds(assignments, 5, seed)
    validation = validation.merge(
        assignments[["well_id", "spatial_fold", "typewell_fold", "branch_group_fold"]],
        on="well_id", how="left", validate="many_to_one",
    )
    validation["stage16_fold"] = validation["stage16_fold"].astype(int)
    wells = sorted(validation["well_id"].astype(str).unique())
    public = pd.read_parquet(
        public_run / "base_oof.parquet", columns=["well_id", "row_index", "y_pred"],
        filters=[("well_id", "in", wells)],
    )
    public["well_id"] = public["well_id"].astype(str)
    public_by_well = {well: frame.sort_values("row_index") for well, frame in public.groupby("well_id")}
    train_dir = args.data_dir.resolve() / "train"

    @lru_cache(maxsize=64)
    def load_well(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train_dir / f"{well_id}__horizontal_well.csv")

    @lru_cache(maxsize=64)
    def load_typewell(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train_dir / f"{well_id}__typewell.csv")

    candidate_config = dict(config.get("candidates", {}))
    feature_config = dict(config.get("features", {}))
    base_name = str(config.get("base_candidate", "top_pf_a130"))
    target_scale = float(config.get("target_scale_ft", 10.0))
    profiles = [dict(value) for value in config.get("profiles", [])]
    correction = dict(config.get("correction", {}))
    rows = []
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
            raise AssertionError("Validation feature schema differs from Stage 30")
        normalized = ((features - feature_mean) / feature_scale).astype(np.float32)
        predictions = target_scale * np.stack([_predict(model, normalized, device) for model in models])
        base = candidates[base_name]
        truth = horizontal["TVT"].to_numpy(float)[outer:]
        row: dict[str, Any] = {
            "cut_id": str(cut.cut_id), "well_id": well_id,
            "requested_fraction": float(cut.requested_fraction),
            **{family: int(getattr(cut, family)) for family in FOLD_FAMILIES},
            "suffix_rows": len(truth), "base_sse": float(np.square(base - truth).sum()),
            "ensemble_spread_mean": float(predictions.std(axis=0).mean()),
        }
        for profile in profiles:
            residual, gate = uncertainty_shrunk_residual(
                predictions, kind=str(profile["kind"]),
                power=float(profile.get("power", 1.0)),
                minimum_agreement=float(profile.get("minimum_agreement", 0.6)),
            )
            prediction = _correction_prediction(
                base, residual, float(profile["weight"]),
                float(correction.get("cap_ft", 8.0)),
                float(correction.get("ramp_rows", 96.0)),
            )
            name = str(profile["name"])
            row[f"candidate_sse_{name}"] = float(np.square(prediction - truth).sum())
            row[f"gate_mean_{name}"] = float(gate.mean())
        rows.append(row)
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
            print(f"TCN uncertainty {position}/{len(validation)} cuts", flush=True)
    report = pd.DataFrame(rows)
    primary = str(config["primary_profile"])
    report["candidate_sse"] = report[f"candidate_sse_{primary}"]
    metrics = _metrics(report)
    bootstrap = _bootstrap(report, int(config.get("validation", {}).get("bootstrap_resamples", 4000)), seed)
    family_reports = {family: _family_report(report, family) for family in FOLD_FAMILIES}
    total_rows = int(report["suffix_rows"].sum())
    base_rmse = float(np.sqrt(report["base_sse"].sum() / total_rows))
    profile_report = []
    for profile in profiles:
        name = str(profile["name"])
        candidate = float(np.sqrt(report[f"candidate_sse_{name}"].sum() / total_rows))
        profile_report.append(
            {
                "profile": name, "kind": str(profile["kind"]),
                "base_rmse": base_rmse, "candidate_rmse": candidate,
                "rmse_delta": candidate - base_rmse,
                "mean_gate": float(report[f"gate_mean_{name}"].mean()),
            }
        )
    validation_config = dict(config.get("validation", {}))
    gates = {
        "hidden_target_invariance": bool(invariance) and all(invariance),
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
    report.to_parquet(output / "uncertainty_cut_report.parquet", index=False)
    summary = {
        "stage31a_complete": True,
        "promoted_to_stage31b_reserved_confirmation": promoted,
        "stage16b_manifest_sha256": expected_hash, "device": str(device),
        "cuts": len(report), "wells": int(report["well_id"].nunique()),
        "ensemble_models": len(models), "primary_profile": primary,
        "base_rmse": base_rmse, "candidate_rmse": metrics["candidate_rmse"],
        "rmse_delta": metrics["rmse_delta"], "bootstrap_95pct": bootstrap,
        "cut_rmse_p90_delta": metrics["cut_rmse_p90_delta"],
        "family_reports": family_reports, "profile_report": profile_report,
        "gates": gates, "reserved_confirmation_used": False,
        "next_step": (
            "Run exactly one Stage 31B audit on the frozen 120 confirmation wells."
            if promoted else
            "Reject TCN uncertainty shrinkage and keep the confirmation reserve sealed."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16), "stage17a_run": str(stage17),
        "public_oof_run": str(public_run), "stage30a_run": str(stage30),
        "validation_run": str(validation_run), "data_dir": str(args.data_dir.resolve()),
        "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()

