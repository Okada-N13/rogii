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
from rogii.models.safe_cutback import apply_cutback_profile, select_cutback_physics
from rogii.models.spatial import make_spatial_blocks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nested visible-prefix physics gate for a strong OOF base")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit-wells", type=int)
    return parser


def _rmse(frame: pd.DataFrame, prediction_column: str, mask: np.ndarray) -> float:
    error = frame.loc[mask, prediction_column].to_numpy(dtype=float) - frame.loc[mask, "y_true"].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(error))))


def _select_nested(
    frame: pd.DataFrame,
    fold_column: str,
    profile_names: list[str],
    minimum_gain: float,
    tolerance: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    prediction = frame["base_y_pred"].to_numpy(dtype=float).copy()
    selections: list[dict[str, object]] = []
    folds = sorted(int(value) for value in frame[fold_column].unique())
    for outer in folds:
        outer_mask = frame[fold_column].to_numpy() == outer
        selection_mask = ~outer_mask
        base_selection = _rmse(frame, "base_y_pred", selection_mask)
        eligible: list[tuple[float, str, float]] = []
        for name in profile_names:
            column = f"pred_{name}"
            candidate_selection = _rmse(frame, column, selection_mask)
            gain = base_selection - candidate_selection
            inner_deltas = []
            for inner in folds:
                if inner == outer:
                    continue
                inner_mask = frame[fold_column].to_numpy() == inner
                inner_deltas.append(_rmse(frame, column, inner_mask) - _rmse(frame, "base_y_pred", inner_mask))
            worst = max(inner_deltas) if inner_deltas else float("inf")
            if gain >= minimum_gain and worst <= tolerance:
                eligible.append((candidate_selection, name, worst))
        if eligible:
            best_rmse, selected, worst = min(eligible)
            prediction[outer_mask] = frame.loc[outer_mask, f"pred_{selected}"].to_numpy(dtype=float)
            gain = base_selection - best_rmse
        else:
            selected, worst, gain = None, None, 0.0
        selections.append(
            {
                "fold": outer,
                "selected_profile": selected,
                "eligible_profiles": len(eligible),
                "selection_gain": float(gain),
                "worst_inner_fold_delta": None if worst is None else float(worst),
            }
        )
    return prediction, selections


def _robust_inference_profile(
    frame: pd.DataFrame,
    profile_names: list[str],
    standard_column: str,
    spatial_column: str,
    minimum_gain: float,
    tolerance: float,
) -> str | None:
    base_rmse = _rmse(frame, "base_y_pred", np.ones(len(frame), dtype=bool))
    eligible: list[tuple[float, str]] = []
    for name in profile_names:
        column = f"pred_{name}"
        candidate_rmse = _rmse(frame, column, np.ones(len(frame), dtype=bool))
        if base_rmse - candidate_rmse < minimum_gain:
            continue
        deltas = []
        for fold_column in [standard_column, spatial_column]:
            for fold in sorted(frame[fold_column].unique()):
                mask = frame[fold_column].to_numpy() == fold
                deltas.append(_rmse(frame, column, mask) - _rmse(frame, "base_y_pred", mask))
        if max(deltas) <= tolerance:
            eligible.append((candidate_rmse, name))
    return min(eligible)[1] if eligible else None


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output_dir = args.artifact_dir.resolve() / args.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = args.base_run / "base_oof.parquet"
    if not base_path.is_file():
        base_path = args.base_run / "oof.parquet"
    base = pd.read_parquet(base_path)
    required = {"id", "well_id", "row_index", "fold", "y_true", "y_pred"}
    missing = sorted(required - set(base.columns))
    if missing:
        raise ValueError(f"Base OOF is missing columns: {missing}")
    wells = base["well_id"].astype(str).drop_duplicates()
    if args.limit_wells is not None:
        wells = wells.iloc[: args.limit_wells]
        base = base[base["well_id"].astype(str).isin(wells)].reset_index(drop=True)
    # Some later-stage OOF files already retain their own upstream base column.
    # The selected run's y_pred is the only baseline for this experiment.
    base = base.drop(columns=["base_y_pred"], errors="ignore").rename(
        columns={"y_pred": "base_y_pred"}
    )

    cutback = config.get("cutback", {})
    profiles = {str(key): dict(value) for key, value in config.get("profiles", {}).items()}
    if not profiles:
        raise ValueError("At least one profile is required")
    records: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    data_dir = args.data_dir.resolve()
    for index, well_id in enumerate(wells.astype(str), 1):
        horizontal = pd.read_csv(data_dir / "train" / f"{well_id}__horizontal_well.csv")
        well_base = base[base["well_id"].astype(str) == well_id].sort_values("row_index").copy()
        row_index = well_base["row_index"].to_numpy(dtype=int)
        if "TVT" in horizontal:
            truth = horizontal.loc[row_index, "TVT"].to_numpy(dtype=float)
            difference = np.abs(truth - well_base["y_true"].to_numpy(dtype=float))
            maximum_difference = float(difference.max(initial=0.0))
            # The public artifact stores target and last_known_tvt as float32.
            # Reconstructing absolute TVT can therefore introduce a few 1e-4 ft.
            if not np.isfinite(difference).all() or maximum_difference > 1e-3:
                raise RuntimeError(
                    "Base OOF truth does not align with horizontal TVT for "
                    f"{well_id}; max_abs_difference={maximum_difference:.8g}"
                )
        physics, report = select_cutback_physics(
            horizontal,
            cut_fractions=[float(value) for value in cutback.get("fractions", [0.55, 0.70, 0.84])],
            degrees=[int(value) for value in cutback.get("degrees", [1, 2, 3])],
            tails=[None if value in {None, "all"} else int(value) for value in cutback.get("tails", [80, 160, 320, "all"])],
            minimum_holdout_rows=int(cutback.get("minimum_holdout_rows", 35)),
        )
        expected_hidden = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
        mapping = dict(zip(expected_hidden.tolist(), physics.tolist()))
        aligned_physics = np.asarray([mapping.get(int(value), np.nan) for value in row_index], dtype=float)
        well_base["physics_y_pred"] = aligned_physics
        md_values = horizontal.loc[row_index, "MD"].to_numpy(dtype=float)
        well_base["evaluation_md"] = md_values
        known = horizontal[horizontal["TVT_input"].notna()]
        anchor_md = float(known["MD"].iloc[-1])
        for name, profile in profiles.items():
            prediction, application = apply_cutback_profile(
                well_base["base_y_pred"].to_numpy(dtype=float),
                aligned_physics,
                md_values - anchor_md,
                report,
                profile,
            )
            well_base[f"pred_{name}"] = prediction.astype(np.float32)
            report[f"{name}_applied"] = bool(application.get("applied", False))
            report[f"{name}_mean_abs_move"] = float(application.get("mean_abs_move", 0.0))
        report.update({"well_id": well_id, "rows": int(len(well_base))})
        reports.append(report)
        records.append(well_base)
        if index % 25 == 0 or index == len(wells):
            print(f"cutback wells: {index}/{len(wells)}", flush=True)

    frame = pd.concat(records, ignore_index=True)
    coordinates = []
    for well_id in wells.astype(str):
        horizontal = pd.read_csv(data_dir / "train" / f"{well_id}__horizontal_well.csv", usecols=["X", "Y", "TVT_input"])
        anchor = horizontal[horizontal["TVT_input"].notna()].iloc[-1]
        coordinates.append({"well_id": well_id, "x": float(anchor["X"]), "y": float(anchor["Y"])})
    coordinate_frame = pd.DataFrame(coordinates)
    coordinate_frame["spatial_fold"] = make_spatial_blocks(
        coordinate_frame,
        min(int(config.get("validation", {}).get("spatial_blocks", 6)), len(coordinate_frame)),
        int(config.get("seed", 42)),
    )
    frame["spatial_fold"] = frame["well_id"].astype(str).map(
        coordinate_frame.set_index("well_id")["spatial_fold"]
    ).astype(np.int16)

    validation = config.get("validation", {})
    names = list(profiles)
    standard_prediction, standard_selections = _select_nested(
        frame,
        "fold",
        names,
        float(validation.get("minimum_selection_gain", 0.02)),
        float(validation.get("inner_fold_tolerance", 0.0)),
    )
    spatial_prediction, spatial_selections = _select_nested(
        frame,
        "spatial_fold",
        names,
        float(validation.get("minimum_selection_gain", 0.02)),
        float(validation.get("inner_fold_tolerance", 0.0)),
    )
    inference_profile = _robust_inference_profile(
        frame,
        names,
        "fold",
        "spatial_fold",
        float(validation.get("minimum_selection_gain", 0.02)),
        float(validation.get("inner_fold_tolerance", 0.0)),
    )

    baseline = frame[
        ["id", "well_id", "row_index", "evaluation_md", "y_true", "base_y_pred", "fold"]
    ].rename(columns={"base_y_pred": "y_pred", "evaluation_md": "MD"})
    candidate = baseline.copy()
    candidate["y_pred"] = standard_prediction
    spatial_baseline = baseline.copy()
    spatial_baseline["fold"] = frame["spatial_fold"].to_numpy()
    spatial_candidate = spatial_baseline.copy()
    spatial_candidate["y_pred"] = spatial_prediction
    gate = evaluate_candidate_gates(
        baseline,
        candidate,
        spatial_baseline,
        spatial_candidate,
        minimum_standard_gain=float(validation.get("minimum_standard_gain", 0.05)),
        minimum_spatial_gain=float(validation.get("minimum_spatial_gain", 0.02)),
        minimum_improved_fold_fraction=float(validation.get("minimum_improved_fold_fraction", 0.8)),
        bootstrap_resamples=int(validation.get("bootstrap_resamples", 2000)),
        seed=int(config.get("seed", 42)),
    )
    gate["inference_profile"] = inference_profile
    gate["promoted"] = bool(gate["promoted"] and inference_profile is not None)
    gate["standard_selections"] = standard_selections
    gate["spatial_selections"] = spatial_selections
    profile_metrics = {}
    for name in names:
        evaluation = baseline.copy()
        evaluation["y_pred"] = frame[f"pred_{name}"].to_numpy(dtype=float)
        profile_metrics[name] = evaluate_predictions(evaluation)[0]
    gate["profile_metrics"] = profile_metrics

    frame.to_parquet(output_dir / "candidate_matrix.parquet", index=False)
    pd.DataFrame(reports).drop(columns=["cut_rows"], errors="ignore").to_parquet(output_dir / "well_reports.parquet", index=False)
    candidate.to_parquet(output_dir / "oof.parquet", index=False)
    spatial_candidate.to_parquet(output_dir / "spatial_oof.parquet", index=False)
    coordinate_frame.to_parquet(output_dir / "spatial_wells.parquet", index=False)
    write_json(output_dir / "gate_summary.json", gate)
    write_json(output_dir / "environment.json", environment_report())
    resolved = dict(config)
    resolved["resolved"] = {
        "base_run": str(args.base_run.resolve()),
        "data_dir": str(data_dir),
        "run_id": args.run_id,
        "limit_wells": args.limit_wells,
    }
    write_yaml(output_dir / "config.yaml", resolved)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"run artifacts: {output_dir}")


if __name__ == "__main__":
    main()
