from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.evaluation.gates import evaluate_candidate_gates
from rogii.models.sequence_features import SEQUENCE_FEATURE_NAMES, SequenceWell, build_sequence_well
from rogii.models.sequence_tcn import predict_tcn, train_tcn_fold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-fit an independent residual TCN")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cutback-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit-wells", type=int)
    return parser


def _rmse(frame: pd.DataFrame, prediction: np.ndarray, mask: np.ndarray) -> float:
    error = np.asarray(prediction)[mask] - frame.loc[mask, "y_true"].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(error))))


def _nested_weights(
    frame: pd.DataFrame,
    residual: np.ndarray,
    weights: list[float],
    minimum_gain: float,
    tolerance: float,
) -> tuple[np.ndarray, list[dict[str, object]], float | None, list[dict[str, float | bool]]]:
    base = frame["base_y_pred"].to_numpy(dtype=float)
    folds = frame["fold"].to_numpy(dtype=int)
    output = base.copy()
    selections = []
    unique_folds = sorted(int(value) for value in np.unique(folds))
    for outer in unique_folds:
        selection = folds != outer
        base_rmse = _rmse(frame, base, selection)
        eligible = []
        for weight in weights:
            prediction = base + weight * residual
            candidate_rmse = _rmse(frame, prediction, selection)
            inner_deltas = [
                _rmse(frame, prediction, folds == inner) - _rmse(frame, base, folds == inner)
                for inner in unique_folds
                if inner != outer
            ]
            gain = base_rmse - candidate_rmse
            worst = max(inner_deltas)
            if gain >= minimum_gain and worst <= tolerance:
                eligible.append((candidate_rmse, weight, worst))
        outer_mask = folds == outer
        if eligible:
            best_rmse, selected_weight, worst = min(eligible)
            output[outer_mask] = base[outer_mask] + selected_weight * residual[outer_mask]
            gain = base_rmse - best_rmse
        else:
            selected_weight, worst, gain = None, None, 0.0
        selections.append(
            {
                "fold": outer,
                "selected_weight": selected_weight,
                "eligible_weights": len(eligible),
                "selection_gain": float(gain),
                "worst_inner_fold_delta": None if worst is None else float(worst),
            }
        )

    full_base_rmse = _rmse(frame, base, np.ones(len(frame), dtype=bool))
    report = []
    robust = []
    for weight in weights:
        prediction = base + weight * residual
        pooled = _rmse(frame, prediction, np.ones(len(frame), dtype=bool))
        deltas = [
            _rmse(frame, prediction, folds == fold) - _rmse(frame, base, folds == fold)
            for fold in unique_folds
        ]
        row = {
            "weight": weight,
            "pooled_rmse": pooled,
            "pooled_gain": full_base_rmse - pooled,
            "worst_fold_delta": max(deltas),
            "eligible": bool(full_base_rmse - pooled >= minimum_gain and max(deltas) <= tolerance),
        }
        report.append(row)
        if row["eligible"]:
            robust.append((pooled, weight))
    inference_weight = min(robust)[1] if robust else None
    return output, selections, inference_weight, report


def _evaluation_frame(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": frame["id"].astype(str),
            "well_id": frame["well_id"].astype(str),
            "MD": frame["MD"].to_numpy(dtype=float),
            "y_true": frame["y_true"].to_numpy(dtype=float),
            "y_pred": np.asarray(prediction, dtype=float),
            "fold": frame["fold"].to_numpy(dtype=np.int16),
        }
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output_dir = args.artifact_dir.resolve() / args.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    matrix = pd.read_parquet(args.cutback_run / "candidate_matrix.parquet")
    well_ids = matrix["well_id"].astype(str).drop_duplicates()
    if args.limit_wells is not None:
        well_ids = well_ids.iloc[: args.limit_wells]
        matrix = matrix[matrix["well_id"].astype(str).isin(well_ids)].reset_index(drop=True)
    data_dir = args.data_dir.resolve()
    wells: list[SequenceWell] = []
    for index, well_id in enumerate(well_ids, 1):
        horizontal = pd.read_csv(data_dir / "train" / f"{well_id}__horizontal_well.csv")
        typewell = pd.read_csv(data_dir / "train" / f"{well_id}__typewell.csv")
        rows = matrix[matrix["well_id"].astype(str) == well_id]
        wells.append(build_sequence_well(horizontal, typewell, rows))
        if index % 25 == 0 or index == len(well_ids):
            print(f"sequence features: {index}/{len(well_ids)}", flush=True)
    del matrix

    folds = sorted({well.fold for well in wells})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"TCN device: {device}; wells={len(wells)}; folds={folds}", flush=True)
    model_config = dict(config.get("model", {}))
    raw_by_well: dict[str, np.ndarray] = {}
    histories = {}
    for fold in folds:
        train_wells = [well for well in wells if well.fold != fold]
        valid_wells = [well for well in wells if well.fold == fold]
        print(f"training fold {fold}: train={len(train_wells)} valid={len(valid_wells)}", flush=True)
        model, mean, scale, history = train_tcn_fold(
            train_wells,
            valid_wells,
            model_config,
            int(config.get("seed", 42)) + fold,
            device,
        )
        predictions = predict_tcn(model, valid_wells, mean, scale, device)
        for well, prediction in zip(valid_wells, predictions, strict=True):
            raw_by_well[well.well_id] = prediction
        histories[str(fold)] = history
        torch.save(
            {
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "mean": mean,
                "scale": scale,
                "feature_names": SEQUENCE_FEATURE_NAMES,
                "config": model_config,
                "fold": fold,
            },
            checkpoint_dir / f"fold_{fold}.pt",
        )
        print(f"fold {fold} best history: {history[-1]}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    records = []
    raw_predictions = []
    for well in wells:
        prediction = raw_by_well[well.well_id]
        raw_predictions.append(prediction)
        records.append(
            pd.DataFrame(
                {
                    "id": well.ids,
                    "well_id": well.well_id,
                    "row_index": well.row_index,
                    "MD": well.md,
                    "fold": well.fold,
                    "y_true": well.y_true,
                    "base_y_pred": well.base_prediction,
                }
            )
        )
    frame = pd.concat(records, ignore_index=True)
    raw_residual = np.concatenate(raw_predictions).astype(float)
    selection = dict(config.get("selection", {}))
    nested_prediction, selections, inference_weight, weight_report = _nested_weights(
        frame,
        raw_residual,
        [float(value) for value in selection.get("weights", [0.05, 0.1, 0.2, 0.35, 0.5])],
        float(selection.get("minimum_selection_gain", 0.02)),
        float(selection.get("inner_fold_tolerance", 0.0)),
    )
    base_prediction = frame["base_y_pred"].to_numpy(dtype=float)
    baseline = _evaluation_frame(frame, base_prediction)
    candidate = _evaluation_frame(frame, nested_prediction)
    validation = dict(config.get("validation", {}))
    gate = evaluate_candidate_gates(
        baseline,
        candidate,
        minimum_standard_gain=float(validation.get("minimum_standard_gain", 0.05)),
        minimum_improved_fold_fraction=float(validation.get("minimum_improved_fold_fraction", 0.8)),
        bootstrap_resamples=int(validation.get("bootstrap_resamples", 2000)),
        seed=int(config.get("seed", 42)),
    )
    gate["inference_weight"] = inference_weight
    gate["promoted_to_spatial_audit"] = bool(gate["promoted"] and inference_weight is not None)
    gate["promoted"] = False
    gate["promotion_note"] = "Stage 9A requires a separate spatial cross-fit before Kaggle integration."
    gate["selections"] = selections
    gate["weight_report"] = weight_report

    frame["raw_tcn_residual"] = raw_residual.astype(np.float32)
    frame["y_pred"] = nested_prediction
    frame.to_parquet(output_dir / "oof.parquet", index=False)
    write_json(output_dir / "gate_summary.json", gate)
    write_json(output_dir / "training_history.json", histories)
    write_json(output_dir / "feature_columns.json", SEQUENCE_FEATURE_NAMES)
    write_json(output_dir / "environment.json", environment_report())
    resolved = dict(config)
    resolved["resolved"] = {
        "cutback_run": str(args.cutback_run.resolve()),
        "data_dir": str(data_dir),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "run_id": args.run_id,
        "limit_wells": args.limit_wells,
        "device": str(device),
    }
    write_yaml(output_dir / "config.yaml", resolved)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"run artifacts: {output_dir}")


if __name__ == "__main__":
    main()
