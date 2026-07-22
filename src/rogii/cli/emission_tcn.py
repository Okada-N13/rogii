from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.data.loading import discover_horizontal_wells, load_horizontal_well, load_typewell, well_id_from_path
from rogii.evaluation.metrics import evaluate_predictions
from rogii.models.emission_features import CANDIDATE_CHANNELS, EmissionSequence, build_emission_sequence, feature_invariance
from rogii.models.emission_tcn import CandidateEmissionTCN, predict_emissions, train_emission_fold
from rogii.models.raw_ncc import offset_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 12B learned 61-state emission TCN")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage11-run", type=Path, required=True)
    parser.add_argument("--stage11c-run", type=Path, required=True)
    parser.add_argument("--stage12a-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--limit-wells", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true", help="Reuse completed fold checkpoints")
    return parser


def _load_wells(data_dir: Path, selected: set[str]):
    horizontal, typewell = {}, {}
    for path in discover_horizontal_wells(data_dir, "train"):
        well_id = well_id_from_path(path)
        if well_id in selected:
            horizontal[well_id] = load_horizontal_well(path)
            typewell[well_id] = load_typewell(path)
    missing = selected - set(horizontal)
    if missing:
        raise FileNotFoundError(f"Missing Stage 12B wells: {sorted(missing)[:3]}")
    return horizontal, typewell


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def _sequence_rows(item: EmissionSequence, logits: np.ndarray, offsets: np.ndarray, raw_temperature: float) -> pd.DataFrame:
    learned = _softmax(logits)
    raw_logits = -item.costs[:, 3].astype(np.float32) / raw_temperature
    raw = _softmax(raw_logits)
    learned_order = np.argsort(-learned, axis=1)
    raw_order = np.argsort(-raw, axis=1)
    target = item.target_state.astype(int)
    learned_rank = 1 + np.sum(learned > learned[np.arange(len(target)), target, None], axis=1)
    raw_rank = 1 + np.sum(raw > raw[np.arange(len(target)), target, None], axis=1)
    learned_expected = learned @ offsets
    learned_map = offsets[np.argmax(learned, axis=1)]
    valid = item.valid
    return pd.DataFrame(
        {
            "id": item.cut_id + "_" + item.row_index.astype(str),
            "well_id": item.well_id,
            "cut_id": item.cut_id,
            "cut_fraction": item.cut_fraction,
            "row_index": item.row_index,
            "MD": item.md,
            "fold": item.fold,
            "spatial_fold": item.spatial_fold,
            "typewell_fold": item.typewell_fold,
            "y_true": item.y_true,
            "surface_y_pred": item.surface_y_pred,
            "true_offset": item.true_offset,
            "emission_valid": valid,
            "raw_rank": raw_rank.astype(np.int16),
            "raw_top5": valid & np.any(raw_order[:, :5] == target[:, None], axis=1),
            "raw_top10": valid & np.any(raw_order[:, :10] == target[:, None], axis=1),
            "raw_nll": -np.log(np.maximum(raw[np.arange(len(target)), target], 1e-12)),
            "learned_rank": learned_rank.astype(np.int16),
            "learned_top5": valid & np.any(learned_order[:, :5] == target[:, None], axis=1),
            "learned_top10": valid & np.any(learned_order[:, :10] == target[:, None], axis=1),
            "learned_nll": -np.log(np.maximum(learned[np.arange(len(target)), target], 1e-12)),
            "learned_entropy": -np.sum(learned * np.log(np.maximum(learned, 1e-12)), axis=1),
            "expected_offset": learned_expected,
            "expected_y_pred": item.surface_y_pred + learned_expected,
            "map_offset": learned_map,
            "map_y_pred": item.surface_y_pred + learned_map,
        }
    )


def _rmse(frame: pd.DataFrame, prediction: str) -> float:
    error = frame[prediction].to_numpy(float) - frame["y_true"].to_numpy(float)
    return float(np.sqrt(np.mean(np.square(error))))


def _rank_report(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    valid = frame["emission_valid"].to_numpy(bool)
    return {
        "valid_fraction": float(valid.mean()),
        "top5_recall": float(frame.loc[valid, f"{prefix}_top5"].mean()),
        "top10_recall": float(frame.loc[valid, f"{prefix}_top10"].mean()),
        "median_rank": float(frame.loc[valid, f"{prefix}_rank"].median()),
        "nll": float(frame.loc[valid, f"{prefix}_nll"].mean()),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    ncc = dict(config.get("ncc", {}))
    model_config = dict(config.get("model", {}))
    validation = dict(config.get("validation", {}))
    stage11 = args.stage11_run.resolve()
    stage11c = args.stage11c_run.resolve()
    stage12a = args.stage12a_run.resolve()
    summary11c = load_config(stage11c / "gate_summary.json")
    summary12a = load_config(stage12a / "benchmark_summary.json")
    if not summary11c.get("promoted_to_stage12") or not summary12a.get("promoted_to_learned_emission"):
        raise RuntimeError("Stage 11C and Stage 12A must both promote this experiment")
    surface = dict(summary11c["selected_inference_parameters"])
    if surface["name"] != validation.get("required_surface_spec", "w075_cap50"):
        raise RuntimeError(f"Unexpected fixed surface: {surface}")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    coefficients = pd.read_parquet(stage11 / "fold_coefficient_oof.parquet")
    wells = sorted(coefficients["well_id"].astype(str).unique())
    if args.limit_wells:
        wells = wells[: args.limit_wells]
        coefficients = coefficients[coefficients["well_id"].astype(str).isin(set(wells))].copy()
    horizontal, typewell = _load_wells(args.data_dir.resolve(), set(wells))
    invariance_rows = []
    for record in coefficients.head(min(6, len(coefficients))).itertuples(index=False):
        passed = feature_invariance(
            record, horizontal[str(record.well_id)], typewell[str(record.well_id)], ncc,
            weight=float(surface["weight"]), correction_cap_ft=float(surface["cap"]),
        )
        invariance_rows.append({"cut_id": str(record.cut_id), "passed": passed})
    hidden_invariance = bool(invariance_rows) and all(row["passed"] for row in invariance_rows)
    if not hidden_invariance:
        raise AssertionError(f"Stage 12B hidden target invariance failed: {invariance_rows}")

    sequences = []
    for index, record in enumerate(coefficients.itertuples(index=False), 1):
        sequences.append(
            build_emission_sequence(
                record, horizontal[str(record.well_id)], typewell[str(record.well_id)], ncc,
                weight=float(surface["weight"]), correction_cap_ft=float(surface["cap"]),
            )
        )
        if index % 100 == 0:
            print(f"built {index}/{len(coefficients)} cost-volume sequences", flush=True)
    offsets = offset_grid(ncc)
    device = _device(args.device)
    print(f"Stage 12B device: {device}; cuts={len(sequences)}; states={len(offsets)}", flush=True)
    oof_rows, histories = [], {}
    present_folds = sorted({item.fold for item in sequences})
    for fold in present_folds:
        train = [item for item in sequences if item.fold != fold]
        valid = [item for item in sequences if item.fold == fold]
        if not train or not valid:
            continue
        checkpoint_path = output / f"fold_{fold}.pt"
        history_path = output / f"fold_{fold}_history.json"
        if args.resume and checkpoint_path.is_file():
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if not np.array_equal(np.asarray(checkpoint["offsets"]), offsets):
                raise RuntimeError(f"Fold {fold} checkpoint offset grid does not match the config")
            model = CandidateEmissionTCN(
                int(checkpoint["n_costs"]), int(checkpoint["n_row_features"]), offsets,
                dict(checkpoint["model_config"]),
            ).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
            print(f"Reusing completed fold checkpoint: {checkpoint_path}", flush=True)
        else:
            model, history = train_emission_fold(
                train, valid, offsets, model_config, int(config.get("seed", 42)) + fold, device
            )
            checkpoint = {
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "offsets": offsets,
                "model_config": model_config,
                "n_costs": int(train[0].costs.shape[1]),
                "n_row_features": int(train[0].row_features.shape[1]),
            }
            torch.save(checkpoint, checkpoint_path)
            write_json(history_path, history)
        histories[str(fold)] = history
        predictions = predict_emissions(model, valid, device)
        oof_rows.extend(
            _sequence_rows(item, logits, offsets, float(validation.get("raw_temperature", 0.25)))
            for item, logits in zip(valid, predictions, strict=True)
        )
        print({"fold": fold, "epochs": len(history), "best_valid_nll": min((row["valid_nll"] for row in history), default=None)}, flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not oof_rows:
        raise RuntimeError("No OOF predictions were produced")
    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_parquet(output / "oof_emission.parquet", index=False)
    raw_report = _rank_report(oof, "raw")
    learned_report = _rank_report(oof, "learned")
    surface_rmse = _rmse(oof, "surface_y_pred")
    expected_rmse = _rmse(oof, "expected_y_pred")
    map_rmse = _rmse(oof, "map_y_pred")
    fold_report = {}
    for fold, frame in oof.groupby("fold"):
        raw_fold = _rank_report(frame, "raw")
        learned_fold = _rank_report(frame, "learned")
        fold_report[f"fold_{fold}"] = {
            "raw_top10": raw_fold["top10_recall"],
            "learned_top10": learned_fold["top10_recall"],
            "top10_delta": learned_fold["top10_recall"] - raw_fold["top10_recall"],
            "nll_delta": learned_fold["nll"] - raw_fold["nll"],
        }
    improved_folds = sum(row["top10_delta"] > 0 for row in fold_report.values())
    spatial_top10 = {f"fold_{int(key)}": float(group.loc[group.emission_valid, "learned_top10"].mean()) for key, group in oof.groupby("spatial_fold")}
    typewell_top10 = {f"fold_{int(key)}": float(group.loc[group.emission_valid, "learned_top10"].mean()) for key, group in oof.groupby("typewell_fold")}
    required_folds = min(int(validation.get("minimum_improved_folds", 4)), len(fold_report))
    gates = {
        "hidden_target_invariance": hidden_invariance,
        "top10_improves_raw": learned_report["top10_recall"] - raw_report["top10_recall"] >= float(validation.get("minimum_top10_gain", 0.02)),
        "top5_improves_raw": learned_report["top5_recall"] > raw_report["top5_recall"],
        "fold_consistency": improved_folds >= required_folds,
        "nll_improves_raw": learned_report["nll"] < raw_report["nll"],
        "finite_expected_alignment": bool(np.isfinite(expected_rmse)),
    }
    promoted = all(gates.values())
    summary = {
        "promoted_to_spatial_emission_audit": promoted,
        "experiment": "stage12b_learned_emission_tcn",
        "device": str(device),
        "surface_spec": surface,
        "n_wells": int(oof.well_id.nunique()),
        "n_cuts": int(oof.cut_id.nunique()),
        "rows": int(len(oof)),
        "offset_states": int(len(offsets)),
        "candidate_channels": list(CANDIDATE_CHANNELS),
        "raw": raw_report,
        "learned": learned_report,
        "top10_gain": learned_report["top10_recall"] - raw_report["top10_recall"],
        "top5_gain": learned_report["top5_recall"] - raw_report["top5_recall"],
        "nll_delta": learned_report["nll"] - raw_report["nll"],
        "surface_rmse": surface_rmse,
        "expected_offset_rmse": expected_rmse,
        "expected_offset_delta": expected_rmse - surface_rmse,
        "map_offset_rmse": map_rmse,
        "map_offset_delta": map_rmse - surface_rmse,
        "improved_folds": f"{improved_folds}/{len(fold_report)}",
        "fold_report": fold_report,
        "spatial_group_top10": spatial_top10,
        "typewell_group_top10": typewell_top10,
        "hidden_target_invariance": invariance_rows,
        "gates": gates,
        "next_step": "Run Stage 12C spatial/typewell cross-fit and K-best lattice." if promoted else "Revise learned emission before path decoding.",
    }
    write_json(output / "training_history.json", histories)
    write_json(output / "gate_summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {"stage11_run": str(stage11), "stage11c_run": str(stage11c), "stage12a_run": str(stage12a), "data_dir": str(args.data_dir.resolve()), "run_id": args.run_id, "limit_wells": args.limit_wells, "resume": args.resume}
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)
    print(f"run artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
