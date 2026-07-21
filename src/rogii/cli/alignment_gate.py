from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.evaluation.gates import evaluate_candidate_gates
from rogii.evaluation.metrics import evaluate_predictions
from rogii.models.alignment_gate import apply_alignment_profile
from rogii.models.spatial import make_spatial_blocks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tail-safe confidence gate for Stage 10 alignment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--alignment-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit-wells", type=int)
    return parser


def _rmse(frame: pd.DataFrame, column: str, mask: np.ndarray) -> float:
    error = frame.loc[mask, column].to_numpy(dtype=float) - frame.loc[mask, "y_true"].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(error))))


def _select_nested(
    frame: pd.DataFrame, fold_column: str, names: list[str], minimum_gain: float, tolerance: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    base = frame["base_y_pred"].to_numpy(dtype=float)
    output = base.copy()
    folds = sorted(int(value) for value in frame[fold_column].unique())
    fold_values = frame[fold_column].to_numpy(dtype=int)
    selections = []
    for outer in folds:
        selection = fold_values != outer
        base_rmse = _rmse(frame, "base_y_pred", selection)
        eligible = []
        for name in names:
            candidate_rmse = _rmse(frame, f"pred_{name}", selection)
            deltas = [
                _rmse(frame, f"pred_{name}", fold_values == inner) - _rmse(frame, "base_y_pred", fold_values == inner)
                for inner in folds if inner != outer
            ]
            gain = base_rmse - candidate_rmse
            worst = max(deltas)
            if gain >= minimum_gain and worst <= tolerance:
                eligible.append((candidate_rmse, name, worst))
        outer_mask = fold_values == outer
        if eligible:
            best, selected, worst = min(eligible, key=lambda value: value[0])
            output[outer_mask] = frame.loc[outer_mask, f"pred_{selected}"].to_numpy(dtype=float)
            gain = base_rmse - best
        else:
            selected, worst, gain = None, None, 0.0
        selections.append({
            "fold": outer, "selected_profile": selected, "eligible_profiles": len(eligible),
            "selection_gain": float(gain), "worst_inner_fold_delta": None if worst is None else float(worst),
        })
    return output, selections


def _robust_profile(
    frame: pd.DataFrame, names: list[str], minimum_gain: float, tolerance: float,
) -> tuple[str | None, list[dict[str, object]]]:
    all_rows = np.ones(len(frame), dtype=bool)
    base = _rmse(frame, "base_y_pred", all_rows)
    report = []
    robust = []
    for name in names:
        pooled = _rmse(frame, f"pred_{name}", all_rows)
        partition_worst = {}
        all_deltas = []
        for fold_column in ["fold", "spatial_fold"]:
            deltas = [
                _rmse(frame, f"pred_{name}", frame[fold_column].to_numpy() == fold)
                - _rmse(frame, "base_y_pred", frame[fold_column].to_numpy() == fold)
                for fold in sorted(frame[fold_column].unique())
            ]
            partition_worst[fold_column] = max(deltas)
            all_deltas.extend(deltas)
        eligible = base - pooled >= minimum_gain and max(all_deltas) <= tolerance
        row = {"profile": name, "pooled_rmse": pooled, "pooled_gain": base - pooled, **partition_worst, "eligible": bool(eligible)}
        report.append(row)
        if eligible:
            robust.append((pooled, name))
    return (min(robust)[1] if robust else None), report


def _evaluation(frame: pd.DataFrame, prediction: np.ndarray, fold_column: str) -> pd.DataFrame:
    return pd.DataFrame({
        "id": frame["id"].astype(str), "well_id": frame["well_id"].astype(str), "MD": frame["MD"].to_numpy(dtype=float),
        "y_true": frame["y_true"].to_numpy(dtype=float), "y_pred": prediction,
        "fold": frame[fold_column].to_numpy(dtype=np.int16),
    })


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output_dir = args.artifact_dir.resolve() / args.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.alignment_run / "oof.parquet")
    diagnostics = pd.read_parquet(args.alignment_run / "well_diagnostics.parquet")
    well_ids = frame["well_id"].astype(str).drop_duplicates()
    if args.limit_wells is not None:
        well_ids = well_ids.iloc[: args.limit_wells]
        frame = frame[frame["well_id"].astype(str).isin(well_ids)].reset_index(drop=True)
        diagnostics = diagnostics[diagnostics["well_id"].astype(str).isin(well_ids)].reset_index(drop=True)
    profiles = {str(name): dict(value) for name, value in config.get("profiles", {}).items()}
    application = {}
    for name, profile in profiles.items():
        frame[f"pred_{name}"], application[name] = apply_alignment_profile(frame, diagnostics, profile)
    coordinates = []
    data_dir = args.data_dir.resolve()
    for well_id in well_ids.astype(str):
        horizontal = pd.read_csv(data_dir / "train" / f"{well_id}__horizontal_well.csv", usecols=["X", "Y", "TVT_input"])
        anchor = horizontal[horizontal["TVT_input"].notna()].iloc[-1]
        coordinates.append({"well_id": well_id, "x": float(anchor["X"]), "y": float(anchor["Y"])})
    spatial = pd.DataFrame(coordinates)
    spatial["spatial_fold"] = make_spatial_blocks(
        spatial,
        min(int(config.get("validation", {}).get("spatial_blocks", 6)), len(spatial)),
        int(config.get("seed", 42)),
    )
    frame["spatial_fold"] = frame["well_id"].astype(str).map(spatial.set_index("well_id")["spatial_fold"]).astype(np.int16)
    selection = dict(config.get("selection", {}))
    names = list(profiles)
    standard_prediction, standard_selections = _select_nested(frame, "fold", names, float(selection.get("minimum_selection_gain", 0.02)), float(selection.get("inner_fold_tolerance", 0.0)))
    spatial_prediction, spatial_selections = _select_nested(frame, "spatial_fold", names, float(selection.get("minimum_selection_gain", 0.02)), float(selection.get("inner_fold_tolerance", 0.0)))
    inference_profile, profile_report = _robust_profile(frame, names, float(selection.get("minimum_selection_gain", 0.02)), float(selection.get("inner_fold_tolerance", 0.0)))
    base_values = frame["base_y_pred"].to_numpy(dtype=float)
    validation = dict(config.get("validation", {}))
    gate = evaluate_candidate_gates(
        _evaluation(frame, base_values, "fold"), _evaluation(frame, standard_prediction, "fold"),
        _evaluation(frame, base_values, "spatial_fold"), _evaluation(frame, spatial_prediction, "spatial_fold"),
        minimum_standard_gain=float(validation.get("minimum_standard_gain", 0.05)),
        minimum_spatial_gain=float(validation.get("minimum_spatial_gain", 0.05)),
        minimum_improved_fold_fraction=float(validation.get("minimum_improved_fold_fraction", 0.8)),
        bootstrap_resamples=int(validation.get("bootstrap_resamples", 2000)), seed=int(config.get("seed", 42)),
    )
    gate["inference_profile"] = inference_profile
    gate["promoted"] = bool(gate["promoted"] and inference_profile is not None)
    gate["standard_selections"] = standard_selections
    gate["spatial_selections"] = spatial_selections
    gate["profile_report"] = profile_report
    gate["profile_application"] = application
    profile_metrics = {}
    for name in names:
        profile_metrics[name] = evaluate_predictions(_evaluation(frame, frame[f"pred_{name}"].to_numpy(dtype=float), "fold"))[0]
    gate["profile_metrics"] = profile_metrics
    frame["y_pred"] = standard_prediction.astype(np.float32)
    frame.to_parquet(output_dir / "oof.parquet", index=False)
    spatial.to_parquet(output_dir / "spatial_wells.parquet", index=False)
    write_json(output_dir / "gate_summary.json", gate)
    write_json(output_dir / "environment.json", environment_report())
    resolved = dict(config)
    resolved["resolved"] = {"alignment_run": str(args.alignment_run.resolve()), "data_dir": str(data_dir), "run_id": args.run_id, "limit_wells": args.limit_wells}
    write_yaml(output_dir / "config.yaml", resolved)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"run artifacts: {output_dir}")


if __name__ == "__main__":
    main()
