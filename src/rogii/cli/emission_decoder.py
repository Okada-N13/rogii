from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.branch_retrieval import _bootstrap, _metrics
from rogii.cli.prefix_router import FOLD_FAMILIES, _family_report
from rogii.cli.strong_base_emission import (
    _device,
    _softmax,
    build_strong_base_sequence,
)
from rogii.cli.trajectory_residual import _typewell_folds
from rogii.config import load_config
from rogii.models.emission_tcn import CandidateEmissionTCN, predict_emissions
from rogii.models.raw_ncc import offset_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 23C nested OOF emission decoder")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--public-oof-run", type=Path, required=True)
    parser.add_argument("--stage23b-run", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def posterior_features(logits: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability = _softmax(logits)
    offsets = np.asarray(offsets, float)
    mean = probability @ offsets
    second = probability @ np.square(offsets)
    standard_deviation = np.sqrt(np.maximum(second - np.square(mean), 0.0))
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1)
    maximum = np.max(probability, axis=1)
    ordered = np.partition(probability, -2, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    map_offset = offsets[np.argmax(probability, axis=1)]
    progress = np.linspace(0.0, 1.0, len(logits), dtype=float)
    summary = np.column_stack(
        [
            mean,
            map_offset,
            standard_deviation,
            entropy,
            maximum,
            margin,
            progress,
            mean * maximum,
            mean / np.maximum(standard_deviation, 1.0),
        ]
    )
    affine = mean[:, None]
    return affine.astype(np.float32), summary.astype(np.float32)


def _fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    alpha: float,
) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler().fit(features, sample_weight=sample_weight)
    transformed = scaler.transform(features)
    model = Ridge(alpha=float(alpha)).fit(transformed, target, sample_weight=sample_weight)
    return scaler, model


def _predict_profile(
    profile: dict[str, Any],
    frame: pd.DataFrame,
    *,
    train_frame: pd.DataFrame | None = None,
) -> np.ndarray:
    kind = str(profile["kind"])
    if kind == "direct":
        return float(profile["weight"]) * frame["posterior_mean"].to_numpy(float)
    feature_columns = ["affine_0"] if kind == "affine" else [
        "summary_0", "summary_1", "summary_2", "summary_3", "summary_4",
        "summary_5", "summary_6", "summary_7", "summary_8",
    ]
    if train_frame is None:
        raise ValueError("Fitted decoder requires training rows")
    training_weight = 1.0 / train_frame.groupby("cut_id")["cut_id"].transform("size").to_numpy(float)
    scaler, model = _fit_ridge(
        train_frame[feature_columns].to_numpy(float),
        train_frame["true_offset"].to_numpy(float),
        training_weight,
        float(profile["alpha"]),
    )
    return model.predict(scaler.transform(frame[feature_columns].to_numpy(float)))


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


def _load_checkpoint(path: Path, offsets: np.ndarray, device: torch.device) -> CandidateEmissionTCN:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not np.array_equal(np.asarray(checkpoint["offsets"]), offsets):
        raise RuntimeError(f"Offset mismatch: {path}")
    model = CandidateEmissionTCN(
        int(checkpoint["n_costs"]),
        int(checkpoint["n_row_features"]),
        offsets,
        dict(checkpoint["model_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    stage16 = args.stage16b_run.resolve()
    stage17 = args.stage17a_run.resolve()
    public_run = args.public_oof_run.resolve()
    stage23b = args.stage23b_run.resolve()
    training_run = args.training_run.resolve()
    validation_run = args.validation_run.resolve()
    summary23b = json.loads((stage23b / "summary.json").read_text(encoding="utf-8"))
    if summary23b.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 23B manifest provenance mismatch")
    if summary23b.get("promoted_to_stage23c") is not False:
        raise AssertionError("Stage 23C is the fixed repair path for rejected Stage 23B")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_parquet(stage17 / "cut_report.parquet")
    training_ids = pd.read_parquet(training_run / "router_cut_report.parquet", columns=["cut_id"])
    validation_ids = pd.read_parquet(validation_run / "confidence_cut_report.parquet", columns=["cut_id"])
    training = cuts[cuts["cut_id"].isin(training_ids["cut_id"])].copy().sort_values("cut_id")
    validation = cuts[cuts["cut_id"].isin(validation_ids["cut_id"])].copy().sort_values("cut_id")
    training_wells = set(training["well_id"].astype(str))
    validation_wells = set(validation["well_id"].astype(str))
    overlap = sorted(training_wells.intersection(validation_wells))
    if overlap:
        raise AssertionError(f"Training/validation well leakage: {overlap[:5]}")
    assignments = pd.read_parquet(stage16 / "well_assignments.parquet")
    assignments["well_id"] = assignments["well_id"].astype(str)
    assignments["typewell_fold"] = _typewell_folds(assignments, 5, seed)
    columns = ["well_id", "spatial_fold", "typewell_fold", "branch_group_fold"]
    training = training.merge(assignments[columns], on="well_id", how="left", validate="many_to_one")
    validation = validation.merge(assignments[columns], on="well_id", how="left", validate="many_to_one")
    for frame in (training, validation):
        frame["stage16_fold"] = frame["stage16_fold"].astype(int)
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
    state_config = dict(config.get("state", {}))

    def build_many(frame: pd.DataFrame, label: str):
        result = []
        for index, record in enumerate(frame.itertuples(index=False), 1):
            well_id, outer = str(record.well_id), int(record.cut_index)
            source = public_by_well[well_id]
            original = int(source["row_index"].min())
            result.append(
                build_strong_base_sequence(
                    record,
                    load_well(well_id),
                    load_typewell(well_id),
                    source["y_pred"].to_numpy(float)[outer - original :],
                    candidate_config,
                    state_config,
                )
            )
            if index % 10 == 0:
                print(f"{label} decoder volumes {index}/{len(frame)} cuts", flush=True)
        return result

    training_sequences = build_many(training, "training")
    validation_sequences = build_many(validation, "validation")
    offsets = offset_grid(state_config)
    device = _device(args.device)
    folds = sorted({item.fold for item in training_sequences})
    training_logits = {}
    validation_logits = [
        np.zeros((len(item.target_state), len(offsets)), np.float32)
        for item in validation_sequences
    ]
    for fold in folds:
        model = _load_checkpoint(stage23b / f"fold_{fold}.pt", offsets, device)
        heldout = [item for item in training_sequences if item.fold == fold]
        heldout_predictions = predict_emissions(model, heldout, device)
        for item, logits in zip(heldout, heldout_predictions, strict=True):
            training_logits[item.cut_id] = logits.astype(np.float32)
        predictions = predict_emissions(model, validation_sequences, device)
        for target, logits in zip(validation_logits, predictions, strict=True):
            target += logits.astype(np.float32) / len(folds)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print({"decoder_fold": fold, "training_oof_cuts": len(heldout)}, flush=True)

    branch_training = dict(zip(training["cut_id"].astype(str), training["branch_group_fold"].astype(int)))
    branch_validation = dict(zip(validation["cut_id"].astype(str), validation["branch_group_fold"].astype(int)))

    def row_frame(sequences, logits_by_sequence, branch_map):
        parts = []
        for item, logits in zip(sequences, logits_by_sequence, strict=True):
            affine, summary = posterior_features(logits, offsets)
            part = pd.DataFrame(
                {
                    "well_id": item.well_id,
                    "cut_id": item.cut_id,
                    "requested_fraction": item.cut_fraction,
                    "stage16_fold": item.fold,
                    "spatial_fold": item.spatial_fold,
                    "typewell_fold": item.typewell_fold,
                    "branch_group_fold": int(branch_map[item.cut_id]),
                    "surface": item.surface_y_pred,
                    "truth": item.y_true,
                    "true_offset": item.true_offset,
                    "posterior_mean": affine[:, 0],
                }
            )
            for index in range(affine.shape[1]):
                part[f"affine_{index}"] = affine[:, index]
            for index in range(summary.shape[1]):
                part[f"summary_{index}"] = summary[:, index]
            parts.append(part)
        return pd.concat(parts, ignore_index=True)

    training_rows = row_frame(
        training_sequences,
        [training_logits[item.cut_id] for item in training_sequences],
        branch_training,
    )
    profiles = [dict(value) for value in config.get("decoder_profiles", [])]
    correction_config = dict(config.get("correction", {}))
    selection_config = dict(config.get("selection", {}))
    nested_reports = []
    nested_predictions = {}
    for profile in profiles:
        name = str(profile["name"])
        predicted_parts = []
        for fold in folds:
            train_part = training_rows[training_rows["stage16_fold"] != fold]
            outer = training_rows[training_rows["stage16_fold"] == fold].copy()
            correction = _predict_profile(profile, outer, train_frame=train_part)
            values = []
            for _, group in outer.groupby("cut_id", sort=False):
                index = group.index
                values.append(
                    pd.Series(
                        _apply_correction(
                            group["surface"].to_numpy(float),
                            correction[outer.index.get_indexer(index)],
                            float(correction_config.get("cap_ft", 12.0)),
                            float(correction_config.get("ramp_rows", 64.0)),
                        ),
                        index=index,
                    )
                )
            outer[f"prediction_{name}"] = pd.concat(values).sort_index()
            predicted_parts.append(outer)
        nested = pd.concat(predicted_parts).sort_index()
        training_rows[f"prediction_{name}"] = nested[f"prediction_{name}"]
        report = _cut_report(training_rows, f"prediction_{name}")
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
                "kind": str(profile["kind"]),
                "rmse": metrics["candidate_rmse"],
                "rmse_delta": metrics["rmse_delta"],
                "improved_folds": metrics["improved_folds"],
                "worst_fold_delta": worst_fold,
                "p90_delta": p90,
                "bootstrap_95pct": bootstrap,
                "eligible": eligible,
            }
        )
        nested_predictions[name] = report
    eligible = [row for row in nested_reports if row["eligible"]]
    selected_name = min(eligible, key=lambda row: (row["rmse"], row["profile"]))["profile"] if eligible else None
    selected_profile = next((profile for profile in profiles if profile["name"] == selected_name), None)

    validation_rows = row_frame(validation_sequences, validation_logits, branch_validation)
    if selected_profile is None:
        validation_rows["prediction"] = validation_rows["surface"]
    else:
        correction = _predict_profile(selected_profile, validation_rows, train_frame=training_rows)
        predictions = []
        for _, group in validation_rows.groupby("cut_id", sort=False):
            index = group.index
            predictions.append(
                pd.Series(
                    _apply_correction(
                        group["surface"].to_numpy(float),
                        correction[validation_rows.index.get_indexer(index)],
                        float(correction_config.get("cap_ft", 12.0)),
                        float(correction_config.get("ramp_rows", 64.0)),
                    ),
                    index=index,
                )
            )
        validation_rows["prediction"] = pd.concat(predictions).sort_index()
    validation_report = _cut_report(validation_rows, "prediction")
    metrics = _metrics(validation_report)
    bootstrap = _bootstrap(
        validation_report, int(config.get("validation", {}).get("bootstrap_resamples", 3000)), seed
    )
    p90 = _p90_delta(validation_report)
    family_reports = {
        family: _family_report(validation_report, family) for family in FOLD_FAMILIES
    }
    validation_config = dict(config.get("validation", {}))
    gates = {
        "training_validation_well_overlap_zero": len(overlap) == 0,
        "training_oof_profile_available": selected_profile is not None,
        "validation_gain": metrics["rmse_delta"]
        <= -float(validation_config.get("minimum_gain", 0.1)),
        "validation_bootstrap": bootstrap[1] < 0,
        "validation_p90_nonworse": p90 <= 0,
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
    training_rows.to_parquet(output / "training_oof_decoder_rows.parquet", index=False)
    validation_rows.to_parquet(output / "validation_decoder_rows.parquet", index=False)
    validation_report.to_parquet(output / "validation_cut_report.parquet", index=False)
    summary = {
        "stage23c_complete": True,
        "promoted_to_stage23d": promoted,
        "stage16b_manifest_sha256": expected_hash,
        "device": str(device),
        "training_cuts": int(training_rows["cut_id"].nunique()),
        "training_wells": int(training_rows["well_id"].nunique()),
        "validation_cuts": int(validation_rows["cut_id"].nunique()),
        "validation_wells": int(validation_rows["well_id"].nunique()),
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
        "next_step": (
            "Run a second disjoint decoder confirmation before full-data training."
            if promoted
            else "Keep the learned ranker, but reject this decoder family."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16),
        "stage17a_run": str(stage17),
        "public_oof_run": str(public_run),
        "stage23b_run": str(stage23b),
        "training_run": str(training_run),
        "validation_run": str(validation_run),
        "data_dir": str(args.data_dir.resolve()),
        "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
