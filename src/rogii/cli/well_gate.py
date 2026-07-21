from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.evaluation.gates import evaluate_candidate_gates
from rogii.models.well_gate import (
    WELL_GATE_FEATURES,
    build_well_gate_table,
    fit_well_gate,
    predict_well_gate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-fit a conditional per-well physics gate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cutback-run", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _row_prediction(
    matrix: pd.DataFrame,
    table: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, int]:
    selected_wells = set(
        table.loc[np.asarray(scores, dtype=float) >= float(threshold), "well_id"].astype(str)
    )
    selected = matrix["well_id"].astype(str).isin(selected_wells).to_numpy()
    prediction = matrix["base_y_pred"].to_numpy(dtype=float).copy()
    prediction[selected] = matrix.loc[selected, "pred_conservative"].to_numpy(dtype=float)
    return prediction, len(selected_wells)


def _rmse(matrix: pd.DataFrame, prediction: np.ndarray, mask: np.ndarray) -> float:
    error = np.asarray(prediction)[mask] - matrix.loc[mask, "y_true"].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(error))))


def _crossfit_scores(
    table: pd.DataFrame,
    fold_column: str,
    model_config: dict,
    seed: int,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    allowed = np.ones(len(table), dtype=bool) if allowed_mask is None else np.asarray(allowed_mask, dtype=bool)
    scores = np.full(len(table), np.nan, dtype=float)
    folds = sorted(int(value) for value in table.loc[allowed, fold_column].unique())
    for fold in folds:
        valid = allowed & (table[fold_column].to_numpy() == fold)
        train = allowed & ~valid
        if int(train.sum()) < 50 or not valid.any():
            continue
        model = fit_well_gate(table, train, model_config, seed + fold)
        scores[valid] = predict_well_gate(model, table.loc[valid])
    return scores


def _nested_predictions(
    matrix: pd.DataFrame,
    table: pd.DataFrame,
    fold_column: str,
    model_config: dict,
    selection_config: dict,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    output = matrix["base_y_pred"].to_numpy(dtype=float).copy()
    selections: list[dict[str, object]] = []
    thresholds = [float(value) for value in selection_config["thresholds"]]
    fold_values = sorted(int(value) for value in table[fold_column].unique())
    table_fold = table.set_index("well_id")[fold_column]
    row_fold = matrix["well_id"].astype(str).map(table_fold).to_numpy(dtype=int)
    base_prediction = matrix["base_y_pred"].to_numpy(dtype=float)
    for outer in fold_values:
        selection_wells = table[fold_column].to_numpy() != outer
        selection_rows = row_fold != outer
        inner_scores = _crossfit_scores(
            table, fold_column, model_config, seed + outer * 100, selection_wells
        )
        eligible: list[tuple[float, float, int, float]] = []
        base_rmse = _rmse(matrix, base_prediction, selection_rows)
        for threshold in thresholds:
            score_values = np.where(np.isfinite(inner_scores), inner_scores, -np.inf)
            prediction, selected_wells = _row_prediction(matrix, table, score_values, threshold)
            candidate_rmse = _rmse(matrix, prediction, selection_rows)
            inner_deltas = []
            for inner in fold_values:
                if inner == outer:
                    continue
                inner_rows = row_fold == inner
                inner_deltas.append(
                    _rmse(matrix, prediction, inner_rows)
                    - _rmse(matrix, matrix["base_y_pred"].to_numpy(dtype=float), inner_rows)
                )
            worst = max(inner_deltas)
            gain = base_rmse - candidate_rmse
            if (
                gain >= float(selection_config.get("minimum_selection_gain", 0.02))
                and worst <= float(selection_config.get("inner_fold_tolerance", 0.0))
                and selected_wells >= int(selection_config.get("minimum_selected_wells", 10))
            ):
                eligible.append((candidate_rmse, threshold, selected_wells, worst))
        outer_table_mask = table[fold_column].to_numpy() == outer
        outer_rows = row_fold == outer
        if eligible:
            best_rmse, threshold, _, worst = min(eligible)
            model = fit_well_gate(table, selection_wells, model_config, seed + outer)
            outer_scores = predict_well_gate(model, table.loc[outer_table_mask])
            selected_outer = set(
                table.loc[outer_table_mask].loc[outer_scores >= threshold, "well_id"].astype(str)
            )
            apply_rows = outer_rows & matrix["well_id"].astype(str).isin(selected_outer).to_numpy()
            output[apply_rows] = matrix.loc[apply_rows, "pred_conservative"].to_numpy(dtype=float)
            selected_count = len(selected_outer)
            selection_gain = base_rmse - best_rmse
        else:
            threshold, worst, selected_count, selection_gain = None, None, 0, 0.0
        selections.append(
            {
                "fold": outer,
                "threshold": threshold,
                "eligible_thresholds": len(eligible),
                "selected_outer_wells": selected_count,
                "selection_gain": float(selection_gain),
                "worst_inner_fold_delta": None if worst is None else float(worst),
            }
        )
    return output, selections


def _robust_threshold(
    matrix: pd.DataFrame,
    table: pd.DataFrame,
    standard_scores: np.ndarray,
    spatial_scores: np.ndarray,
    selection_config: dict,
) -> tuple[float | None, list[dict[str, object]]]:
    reports = []
    base = matrix["base_y_pred"].to_numpy(dtype=float)
    all_rows = np.ones(len(matrix), dtype=bool)
    base_rmse = _rmse(matrix, base, all_rows)
    table_index = table.set_index("well_id")
    candidates = []
    for threshold in [float(value) for value in selection_config["thresholds"]]:
        row = {"threshold": threshold}
        eligible = True
        total_rmse = 0.0
        for label, scores, fold_column in [
            ("standard", standard_scores, "fold"),
            ("spatial", spatial_scores, "spatial_fold"),
        ]:
            prediction, selected = _row_prediction(matrix, table, scores, threshold)
            pooled = _rmse(matrix, prediction, all_rows)
            row[f"{label}_rmse"] = pooled
            row[f"{label}_gain"] = base_rmse - pooled
            row[f"{label}_selected_wells"] = selected
            total_rmse += pooled
            fold_map = table_index[fold_column]
            row_folds = matrix["well_id"].astype(str).map(fold_map).to_numpy(dtype=int)
            deltas = [
                _rmse(matrix, prediction, row_folds == fold)
                - _rmse(matrix, base, row_folds == fold)
                for fold in sorted(np.unique(row_folds))
            ]
            row[f"{label}_worst_fold_delta"] = max(deltas)
            eligible &= (
                base_rmse - pooled >= float(selection_config.get("minimum_selection_gain", 0.02))
                and max(deltas) <= float(selection_config.get("inner_fold_tolerance", 0.0))
                and selected >= int(selection_config.get("minimum_selected_wells", 10))
            )
        row["eligible"] = bool(eligible)
        reports.append(row)
        if eligible:
            candidates.append((total_rmse, threshold))
    return (min(candidates)[1] if candidates else None), reports


def _evaluation_frame(matrix: pd.DataFrame, prediction: np.ndarray, fold: np.ndarray) -> pd.DataFrame:
    md_column = "evaluation_md" if "evaluation_md" in matrix else "MD"
    return pd.DataFrame(
        {
            "id": matrix["id"].astype(str),
            "well_id": matrix["well_id"].astype(str),
            "MD": matrix[md_column].to_numpy(dtype=float),
            "y_true": matrix["y_true"].to_numpy(dtype=float),
            "y_pred": np.asarray(prediction, dtype=float),
            "fold": np.asarray(fold, dtype=np.int16),
        }
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output_dir = args.artifact_dir.resolve() / args.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_parquet(args.cutback_run / "candidate_matrix.parquet")
    reports = pd.read_parquet(args.cutback_run / "well_reports.parquet")
    table = build_well_gate_table(matrix, reports)
    model_config = dict(config.get("model", {}))
    selection_config = dict(config.get("selection", {}))
    seed = int(config.get("seed", 42))

    standard_prediction, standard_selections = _nested_predictions(
        matrix, table, "fold", model_config, selection_config, seed
    )
    spatial_prediction, spatial_selections = _nested_predictions(
        matrix, table, "spatial_fold", model_config, selection_config, seed + 10_000
    )
    standard_scores = _crossfit_scores(table, "fold", model_config, seed + 20_000)
    spatial_scores = _crossfit_scores(table, "spatial_fold", model_config, seed + 30_000)
    inference_threshold, threshold_report = _robust_threshold(
        matrix, table, standard_scores, spatial_scores, selection_config
    )

    fold_map = table.set_index("well_id")["fold"]
    spatial_map = table.set_index("well_id")["spatial_fold"]
    standard_fold = matrix["well_id"].astype(str).map(fold_map).to_numpy(dtype=int)
    spatial_fold = matrix["well_id"].astype(str).map(spatial_map).to_numpy(dtype=int)
    base = matrix["base_y_pred"].to_numpy(dtype=float)
    baseline = _evaluation_frame(matrix, base, standard_fold)
    candidate = _evaluation_frame(matrix, standard_prediction, standard_fold)
    spatial_baseline = _evaluation_frame(matrix, base, spatial_fold)
    spatial_candidate = _evaluation_frame(matrix, spatial_prediction, spatial_fold)
    validation = dict(config.get("validation", {}))
    gate = evaluate_candidate_gates(
        baseline,
        candidate,
        spatial_baseline,
        spatial_candidate,
        minimum_standard_gain=float(validation.get("minimum_standard_gain", 0.03)),
        minimum_spatial_gain=float(validation.get("minimum_spatial_gain", 0.02)),
        minimum_improved_fold_fraction=float(validation.get("minimum_improved_fold_fraction", 0.8)),
        bootstrap_resamples=int(validation.get("bootstrap_resamples", 2000)),
        seed=seed,
    )
    gate["inference_threshold"] = inference_threshold
    gate["promoted"] = bool(gate["promoted"] and inference_threshold is not None)
    gate["standard_selections"] = standard_selections
    gate["spatial_selections"] = spatial_selections
    gate["threshold_report"] = threshold_report

    full_model = fit_well_gate(table, np.ones(len(table), dtype=bool), model_config, seed + 40_000)
    with (output_dir / "well_gate_full.pkl").open("wb") as handle:
        pickle.dump(full_model, handle)
    table.to_parquet(output_dir / "well_gate_table.parquet", index=False)
    candidate.to_parquet(output_dir / "oof.parquet", index=False)
    spatial_candidate.to_parquet(output_dir / "spatial_oof.parquet", index=False)
    write_json(output_dir / "gate_summary.json", gate)
    write_json(output_dir / "feature_columns.json", WELL_GATE_FEATURES)
    write_json(
        output_dir / "model_manifest.json",
        {
            "schema_version": "rogii_stage8b_well_gate_v1",
            "base": "stage8_conservative_physics_candidate",
            "model": "well_gate_full.pkl",
            "features": "feature_columns.json",
            "inference_threshold": inference_threshold,
            "promoted": gate["promoted"],
        },
    )
    write_json(output_dir / "environment.json", environment_report())
    resolved = dict(config)
    resolved["resolved"] = {
        "cutback_run": str(args.cutback_run.resolve()),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "run_id": args.run_id,
    }
    write_yaml(output_dir / "config.yaml", resolved)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"run artifacts: {output_dir}")


if __name__ == "__main__":
    main()
