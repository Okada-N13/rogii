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
from rogii.cli.prefix_router import FOLD_FAMILIES, _family_report, build_candidates
from rogii.cli.strong_base_ncc import strong_base_costs
from rogii.cli.trajectory_residual import _typewell_folds
from rogii.config import load_config
from rogii.models.emission_features import EmissionSequence
from rogii.models.emission_tcn import predict_emissions, train_emission_fold
from rogii.models.raw_ncc import offset_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 23B disjoint learned strong-base emission")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--public-oof-run", type=Path, required=True)
    parser.add_argument("--stage23a-run", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit-training-cuts", type=int)
    parser.add_argument("--limit-validation-cuts", type=int)
    return parser


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, float) - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def build_strong_base_sequence(
    record: Any,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    public_prediction: np.ndarray,
    candidate_config: dict[str, Any],
    state_config: dict[str, Any],
) -> EmissionSequence:
    outer = int(record.cut_index)
    base_name = str(state_config.get("base_candidate", "top_pf_a130"))
    candidates = build_candidates(
        horizontal, typewell, outer, public_prediction, candidate_config
    )
    base = candidates[base_name]
    positions, surface, offsets, cost_map = strong_base_costs(
        horizontal, typewell, outer, base, state_config
    )
    cost_channels = np.stack(
        [cost_map[name] for name in ("ncc_w5", "ncc_w13", "ncc_w25", "ncc_mix")], axis=1
    )
    suffix = horizontal.iloc[positions]
    md = suffix["MD"].to_numpy(float)
    gr = pd.to_numeric(suffix["GR"], errors="coerce").to_numpy(float)
    finite_gr = gr[np.isfinite(gr)]
    center = float(np.median(finite_gr)) if len(finite_gr) else 0.0
    spread = float(1.4826 * np.median(np.abs(finite_gr - center))) if len(finite_gr) else 1.0
    spread = max(spread, 1.0)
    gr_z = np.nan_to_num((gr - center) / spread, nan=0.0).clip(-6.0, 6.0)
    type_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(float)
    type_gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(float)
    finite_type = np.isfinite(type_tvt) & np.isfinite(type_gr)
    type_tvt, type_gr = type_tvt[finite_type], type_gr[finite_type]
    order = np.argsort(type_tvt)
    type_tvt, type_gr = type_tvt[order], type_gr[order]
    candidate_tvt = surface[:, None] + offsets[None, :]
    candidate_gr = np.interp(candidate_tvt.ravel(), type_tvt, type_gr).reshape(candidate_tvt.shape)
    candidate_gr_z = np.clip((candidate_gr - center) / spread, -6.0, 6.0)
    candidate_valid = (
        (candidate_tvt >= type_tvt[0]) & (candidate_tvt <= type_tvt[-1])
    ).astype(float)
    paired = np.stack(
        [
            np.broadcast_to(gr_z[:, None], candidate_gr_z.shape),
            candidate_gr_z,
            candidate_valid,
        ],
        axis=1,
    )
    costs = np.concatenate([cost_channels, paired], axis=1).astype(np.float32)
    suffix_offset = positions - outer
    selector_delta = (candidates["selector_a130"][suffix_offset] - surface) / 10.0
    public_delta = (candidates["public_oof"][suffix_offset] - surface) / 10.0
    horizon = md - float(md[0])
    horizon_scale = max(float(np.max(horizon)), 1.0)
    slope = np.gradient(surface, md) if len(md) > 1 else np.zeros(len(md))
    row_features = np.column_stack(
        [
            horizon / horizon_scale,
            np.full(len(md), float(record.requested_fraction)),
            np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0).clip(-2.0, 2.0),
            gr_z,
            selector_delta,
            public_delta,
        ]
    ).astype(np.float32)
    truth = horizontal.iloc[positions]["TVT"].to_numpy(float)
    true_offset = truth - surface
    target = np.argmin(np.abs(offsets[None, :] - true_offset[:, None]), axis=1)
    true_cost = cost_map["ncc_mix"][np.arange(len(target)), target]
    valid = (
        np.isfinite(true_offset)
        & (true_offset >= offsets[0])
        & (true_offset <= offsets[-1])
        & (true_cost < 2.999)
    )
    return EmissionSequence(
        cut_id=str(record.cut_id),
        well_id=str(record.well_id),
        fold=int(record.stage16_fold),
        spatial_fold=int(record.spatial_fold),
        typewell_fold=int(record.typewell_fold),
        cut_fraction=float(record.requested_fraction),
        row_index=positions.astype(np.int64),
        md=md.astype(np.float32),
        costs=costs.astype(np.float16),
        row_features=row_features,
        surface_y_pred=surface.astype(np.float32),
        target_state=target.astype(np.int16),
        valid=valid,
        true_offset=true_offset.astype(np.float32),
        y_true=truth.astype(np.float32),
    )


def _rank_values(
    sequence: EmissionSequence,
    logits: np.ndarray,
    offsets: np.ndarray,
    raw_temperature: float,
) -> pd.DataFrame:
    learned = _softmax(logits)
    raw = _softmax(-sequence.costs[:, 3].astype(np.float32) / raw_temperature)
    target = sequence.target_state.astype(int)
    valid = sequence.valid
    learned_rank = 1 + np.sum(
        learned > learned[np.arange(len(target)), target, None], axis=1
    )
    raw_rank = 1 + np.sum(raw > raw[np.arange(len(target)), target, None], axis=1)
    learned_order = np.argsort(-learned, axis=1)
    raw_order = np.argsort(-raw, axis=1)
    return pd.DataFrame(
        {
            "well_id": sequence.well_id,
            "cut_id": sequence.cut_id,
            "requested_fraction": sequence.cut_fraction,
            "stage16_fold": sequence.fold,
            "spatial_fold": sequence.spatial_fold,
            "typewell_fold": sequence.typewell_fold,
            "emission_valid": valid,
            "raw_rank": raw_rank,
            "learned_rank": learned_rank,
            "raw_top5": valid & np.any(raw_order[:, :5] == target[:, None], axis=1),
            "raw_top10": valid & np.any(raw_order[:, :10] == target[:, None], axis=1),
            "learned_top5": valid & np.any(learned_order[:, :5] == target[:, None], axis=1),
            "learned_top10": valid & np.any(learned_order[:, :10] == target[:, None], axis=1),
            "raw_nll": -np.log(np.maximum(raw[np.arange(len(target)), target], 1e-12)),
            "learned_nll": -np.log(
                np.maximum(learned[np.arange(len(target)), target], 1e-12)
            ),
            "surface": sequence.surface_y_pred,
            "truth": sequence.y_true,
            "expected_offset": learned @ np.asarray(offsets, float),
        }
    )


def _rank_report(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    valid = frame["emission_valid"].to_numpy(bool)
    return {
        "valid_fraction": float(valid.mean()),
        "top5_recall": float(frame.loc[valid, f"{prefix}_top5"].mean()),
        "top10_recall": float(frame.loc[valid, f"{prefix}_top10"].mean()),
        "median_rank": float(frame.loc[valid, f"{prefix}_rank"].median()),
        "nll": float(frame.loc[valid, f"{prefix}_nll"].mean()),
    }


def _improved_groups(frame: pd.DataFrame, column: str) -> tuple[int, dict[str, dict[str, float]]]:
    report = {}
    for key, group in frame.groupby(column, sort=True):
        raw = _rank_report(group, "raw")
        learned = _rank_report(group, "learned")
        report[str(key)] = {
            "raw_top10": raw["top10_recall"],
            "learned_top10": learned["top10_recall"],
            "delta": learned["top10_recall"] - raw["top10_recall"],
        }
    return sum(value["delta"] > 0 for value in report.values()), report


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    stage16 = args.stage16b_run.resolve()
    stage17 = args.stage17a_run.resolve()
    public_run = args.public_oof_run.resolve()
    stage23a = args.stage23a_run.resolve()
    training_run = args.training_run.resolve()
    validation_run = args.validation_run.resolve()
    summary23a = json.loads((stage23a / "summary.json").read_text(encoding="utf-8"))
    if summary23a.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 23A manifest provenance mismatch")
    if not summary23a.get("promoted_to_stage23b_learned_emission"):
        raise AssertionError("Stage 23A did not promote learned emission")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_parquet(stage17 / "cut_report.parquet")
    training_ids = pd.read_parquet(training_run / "router_cut_report.parquet", columns=["cut_id"])
    validation_ids = pd.read_parquet(validation_run / "confidence_cut_report.parquet", columns=["cut_id"])
    training = cuts[cuts["cut_id"].isin(training_ids["cut_id"])].copy().sort_values("cut_id")
    validation = cuts[cuts["cut_id"].isin(validation_ids["cut_id"])].copy().sort_values("cut_id")
    if args.limit_training_cuts is not None:
        training = training.head(int(args.limit_training_cuts)).copy()
    if args.limit_validation_cuts is not None:
        validation = validation.head(int(args.limit_validation_cuts)).copy()
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
    branch_by_cut = dict(zip(validation["cut_id"].astype(str), validation["branch_group_fold"].astype(int)))
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

    def build_many(frame: pd.DataFrame, label: str) -> list[EmissionSequence]:
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
                print(f"{label} cost volumes {index}/{len(frame)} cuts", flush=True)
        return result

    train_sequences = build_many(training, "training")
    valid_sequences = build_many(validation, "validation")
    if not train_sequences or not valid_sequences:
        raise RuntimeError("No strong-base emission sequences")
    invariance = []
    for record in validation.head(min(6, len(validation))).itertuples(index=False):
        well_id, outer = str(record.well_id), int(record.cut_index)
        horizontal = load_well(well_id)
        source = public_by_well[well_id]
        original = int(source["row_index"].min())
        public_prediction = source["y_pred"].to_numpy(float)[outer - original :]
        first = build_strong_base_sequence(
            record, horizontal, load_typewell(well_id), public_prediction,
            candidate_config, state_config,
        )
        changed = horizontal.copy()
        changed.loc[changed.index >= outer, "TVT"] += 9999.0
        second = build_strong_base_sequence(
            record, changed, load_typewell(well_id), public_prediction,
            candidate_config, state_config,
        )
        invariance.append(
            np.array_equal(first.costs, second.costs)
            and np.array_equal(first.row_features, second.row_features)
            and np.array_equal(first.surface_y_pred, second.surface_y_pred)
        )
    if not invariance or not all(invariance):
        raise AssertionError("Stage 23B hidden-target invariance failed")

    offsets = offset_grid(state_config)
    device = _device(args.device)
    model_config = dict(config.get("model", {}))
    ensemble_logits = [np.zeros((len(item.target_state), len(offsets)), np.float32) for item in valid_sequences]
    histories = {}
    folds = sorted({item.fold for item in train_sequences})
    for fold in folds:
        train_fold = [item for item in train_sequences if item.fold != fold]
        internal_valid = [item for item in train_sequences if item.fold == fold]
        model, history = train_emission_fold(
            train_fold, internal_valid, offsets, model_config, seed + fold, device
        )
        predictions = predict_emissions(model, valid_sequences, device)
        for target, values in zip(ensemble_logits, predictions, strict=True):
            target += values.astype(np.float32) / len(folds)
        checkpoint = {
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "offsets": offsets,
            "model_config": model_config,
            "n_costs": int(train_fold[0].costs.shape[1]),
            "n_row_features": int(train_fold[0].row_features.shape[1]),
        }
        torch.save(checkpoint, output / f"fold_{fold}.pt")
        write_json(output / f"fold_{fold}_history.json", history)
        histories[str(fold)] = history
        print(
            {
                "training_fold": fold,
                "epochs": len(history),
                "best_internal_top10": max((row["valid_top10"] for row in history), default=None),
            },
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw_temperature = float(config.get("validation", {}).get("raw_temperature", 0.25))
    parts = [
        _rank_values(item, logits, offsets, raw_temperature)
        for item, logits in zip(valid_sequences, ensemble_logits, strict=True)
    ]
    rows = pd.concat(parts, ignore_index=True)
    rows["branch_group_fold"] = rows["cut_id"].map(branch_by_cut).astype(int)
    raw_report = _rank_report(rows, "raw")
    learned_report = _rank_report(rows, "learned")
    correction = dict(config.get("correction", {}))
    weights = [float(value) for value in correction.get("diagnostic_weights", [0.5])]
    primary_weight = float(correction.get("primary_weight", 0.5))
    if primary_weight not in weights:
        weights.append(primary_weight)
    cut_rows = []
    for cut_id, group in rows.groupby("cut_id", sort=True):
        first = group.iloc[0]
        step = np.arange(1, len(group) + 1, dtype=float)
        ramp = 1.0 - np.exp(-step / max(float(correction.get("ramp_rows", 64.0)), 1.0))
        item: dict[str, Any] = {
            "cut_id": cut_id,
            "well_id": str(first["well_id"]),
            "requested_fraction": float(first["requested_fraction"]),
            **{family: int(first[family]) for family in FOLD_FAMILIES},
            "suffix_rows": len(group),
            "base_sse": float(np.square(group["surface"] - group["truth"]).sum()),
        }
        for weight in weights:
            move = np.clip(
                weight * ramp * group["expected_offset"].to_numpy(float),
                -float(correction.get("cap_ft", 12.0)),
                float(correction.get("cap_ft", 12.0)),
            )
            prediction = group["surface"].to_numpy(float) + move
            item[f"candidate_sse_w{int(round(weight * 1000)):03d}"] = float(
                np.square(prediction - group["truth"].to_numpy(float)).sum()
            )
        cut_rows.append(item)
    cut_report = pd.DataFrame.from_records(cut_rows)
    primary_column = f"candidate_sse_w{int(round(primary_weight * 1000)):03d}"
    cut_report["candidate_sse"] = cut_report[primary_column]
    metrics = _metrics(cut_report)
    bootstrap = _bootstrap(
        cut_report, int(config.get("validation", {}).get("bootstrap_resamples", 3000)), seed
    )
    base_well = cut_report.groupby("well_id").agg(sse=("base_sse", "sum"), rows=("suffix_rows", "sum"))
    candidate_well = cut_report.groupby("well_id").agg(
        sse=("candidate_sse", "sum"), rows=("suffix_rows", "sum")
    )
    p90_delta = float(
        np.sqrt(candidate_well.sse / candidate_well.rows).quantile(0.9)
        - np.sqrt(base_well.sse / base_well.rows).quantile(0.9)
    )
    family_reports = {family: _family_report(cut_report, family) for family in FOLD_FAMILIES}
    weight_report = []
    total_rows = int(cut_report["suffix_rows"].sum())
    base_rmse = float(np.sqrt(cut_report["base_sse"].sum() / total_rows))
    for weight in sorted(weights):
        column = f"candidate_sse_w{int(round(weight * 1000)):03d}"
        rmse = float(np.sqrt(cut_report[column].sum() / total_rows))
        weight_report.append(
            {"weight": weight, "base_rmse": base_rmse, "candidate_rmse": rmse, "rmse_delta": rmse - base_rmse}
        )
    group_counts, group_reports = {}, {}
    for family in (*FOLD_FAMILIES, "requested_fraction"):
        count, report = _improved_groups(rows, family)
        group_counts[family], group_reports[family] = count, report
    validation_config = dict(config.get("validation", {}))
    gates = {
        "hidden_target_invariance": all(invariance),
        "training_validation_well_overlap_zero": len(overlap) == 0,
        "top10_gain": learned_report["top10_recall"] - raw_report["top10_recall"]
        >= float(validation_config.get("minimum_top10_gain", 0.1)),
        "top5_gain": learned_report["top5_recall"] - raw_report["top5_recall"]
        >= float(validation_config.get("minimum_top5_gain", 0.1)),
        "nll_gain": learned_report["nll"] < raw_report["nll"],
        "primary_rmse_gain": metrics["rmse_delta"]
        <= -float(validation_config.get("minimum_rmse_gain", 0.1)),
        "bootstrap_upper_below_zero": bootstrap[1] < 0,
        "well_p90_nonworse": p90_delta <= 0,
        "standard_rank_consistency": group_counts["stage16_fold"]
        >= int(validation_config.get("minimum_standard_rank_folds", 4)),
        "spatial_rank_consistency": group_counts["spatial_fold"]
        >= int(validation_config.get("minimum_spatial_rank_folds", 4)),
        "typewell_rank_consistency": group_counts["typewell_fold"]
        >= int(validation_config.get("minimum_typewell_rank_folds", 4)),
        "branch_rank_consistency": group_counts["branch_group_fold"]
        >= int(validation_config.get("minimum_branch_rank_folds", 4)),
        "fraction_rank_consistency": group_counts["requested_fraction"]
        >= int(validation_config.get("minimum_fraction_rank_groups", 3)),
    }
    promoted = bool(all(gates.values()))
    rows.to_parquet(output / "validation_emission_rows.parquet", index=False)
    cut_report.to_parquet(output / "validation_cut_report.parquet", index=False)
    summary = {
        "stage23b_complete": True,
        "promoted_to_stage23c": promoted,
        "stage16b_manifest_sha256": expected_hash,
        "device": str(device),
        "training_cuts": len(train_sequences),
        "training_wells": len(training_wells),
        "validation_cuts": len(valid_sequences),
        "validation_wells": len(validation_wells),
        "training_validation_well_overlap": overlap,
        "ensemble_models": len(folds),
        "rows": len(rows),
        "raw": raw_report,
        "learned": learned_report,
        "top10_gain": learned_report["top10_recall"] - raw_report["top10_recall"],
        "top5_gain": learned_report["top5_recall"] - raw_report["top5_recall"],
        "nll_delta": learned_report["nll"] - raw_report["nll"],
        "base_rmse": metrics["base_rmse"],
        "candidate_rmse": metrics["candidate_rmse"],
        "rmse_delta": metrics["rmse_delta"],
        "well_p90_delta": p90_delta,
        "bootstrap_95pct": bootstrap,
        "weight_report": weight_report,
        "group_improved_counts": group_counts,
        "group_reports": group_reports,
        "gates": gates,
        "next_step": (
            "Run a second disjoint learned-emission confirmation and decoder audit."
            if promoted
            else "Revise learned emission without using the fixed validation targets."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "training_history.json", histories)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16),
        "stage17a_run": str(stage17),
        "public_oof_run": str(public_run),
        "stage23a_run": str(stage23a),
        "training_run": str(training_run),
        "validation_run": str(validation_run),
        "data_dir": str(args.data_dir.resolve()),
        "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
