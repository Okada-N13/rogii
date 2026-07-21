from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.evaluation.gates import evaluate_candidate_gates
from rogii.models.multiscale_alignment import predict_multiscale_alignment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nested multi-scale GR/typewell alignment audit")
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


def _process_well(
    well_id: str,
    rows: pd.DataFrame,
    data_dir: Path,
    branches: dict[str, dict[str, object]],
) -> tuple[str, dict[str, np.ndarray], list[dict[str, object]]]:
    horizontal = pd.read_csv(data_dir / "train" / f"{well_id}__horizontal_well.csv")
    typewell = pd.read_csv(data_dir / "train" / f"{well_id}__typewell.csv")
    prediction = rows[["row_index", "base_y_pred"]].copy()
    corrections: dict[str, np.ndarray] = {}
    diagnostics = []
    for name, branch in branches.items():
        correction, report = predict_multiscale_alignment(horizontal, typewell, prediction, branch)
        corrections[name] = correction.astype(np.float32)
        diagnostics.append({"well_id": well_id, "branch": name, **report})
    return well_id, corrections, diagnostics


def _nested_specs(
    frame: pd.DataFrame,
    specs: list[tuple[str, float]],
    minimum_gain: float,
    tolerance: float,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object] | None, list[dict[str, object]]]:
    base = frame["base_y_pred"].to_numpy(dtype=float)
    folds = frame["fold"].to_numpy(dtype=int)
    unique_folds = sorted(int(value) for value in np.unique(folds))
    output = base.copy()
    candidates = {
        (branch, weight): base + weight * frame[f"correction_{branch}"].to_numpy(dtype=float)
        for branch, weight in specs
    }
    selections = []
    for outer in unique_folds:
        selection_mask = folds != outer
        base_selection = _rmse(frame, base, selection_mask)
        eligible = []
        for spec, prediction in candidates.items():
            candidate_rmse = _rmse(frame, prediction, selection_mask)
            inner_deltas = [
                _rmse(frame, prediction, folds == inner) - _rmse(frame, base, folds == inner)
                for inner in unique_folds if inner != outer
            ]
            gain = base_selection - candidate_rmse
            worst = max(inner_deltas)
            if gain >= minimum_gain and worst <= tolerance:
                eligible.append((candidate_rmse, spec, worst))
        outer_mask = folds == outer
        if eligible:
            best_rmse, selected, worst = min(eligible, key=lambda value: value[0])
            output[outer_mask] = candidates[selected][outer_mask]
            selection_gain = base_selection - best_rmse
            selected_spec = {"branch": selected[0], "weight": selected[1]}
        else:
            selected_spec, worst, selection_gain = None, None, 0.0
        selections.append({
            "fold": outer,
            "selected_spec": selected_spec,
            "eligible_specs": len(eligible),
            "selection_gain": float(selection_gain),
            "worst_inner_fold_delta": None if worst is None else float(worst),
        })

    all_rows = np.ones(len(frame), dtype=bool)
    full_base = _rmse(frame, base, all_rows)
    report = []
    robust = []
    for spec, prediction in candidates.items():
        pooled = _rmse(frame, prediction, all_rows)
        deltas = [
            _rmse(frame, prediction, folds == fold) - _rmse(frame, base, folds == fold)
            for fold in unique_folds
        ]
        eligible = full_base - pooled >= minimum_gain and max(deltas) <= tolerance
        row = {
            "branch": spec[0], "weight": spec[1], "pooled_rmse": pooled,
            "pooled_gain": full_base - pooled, "worst_fold_delta": max(deltas),
            "eligible": bool(eligible),
        }
        report.append(row)
        if eligible:
            robust.append((pooled, spec))
    inference = None
    if robust:
        selected = min(robust, key=lambda value: value[0])[1]
        inference = {"branch": selected[0], "weight": selected[1]}
    return output, selections, inference, report


def _evaluation_frame(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "id": frame["id"].astype(str), "well_id": frame["well_id"].astype(str),
        "MD": frame["MD"].to_numpy(dtype=float), "y_true": frame["y_true"].to_numpy(dtype=float),
        "y_pred": np.asarray(prediction, dtype=float), "fold": frame["fold"].to_numpy(dtype=np.int16),
    })


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output_dir = args.artifact_dir.resolve() / args.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_parquet(args.cutback_run / "candidate_matrix.parquet")
    required = {"id", "well_id", "row_index", "fold", "y_true", "base_y_pred"}
    missing = sorted(required - set(matrix.columns))
    if missing:
        raise ValueError(f"Candidate matrix is missing columns: {missing}")
    well_ids = matrix["well_id"].astype(str).drop_duplicates()
    if args.limit_wells is not None:
        well_ids = well_ids.iloc[: args.limit_wells]
        matrix = matrix[matrix["well_id"].astype(str).isin(well_ids)].reset_index(drop=True)
    if "evaluation_md" in matrix:
        matrix["MD"] = matrix["evaluation_md"].to_numpy(dtype=float)
    elif "MD" not in matrix:
        matrix["MD"] = matrix["row_index"].to_numpy(dtype=float)
    branches = {str(name): dict(value) for name, value in config.get("branches", {}).items()}
    if not branches:
        raise ValueError("At least one alignment branch is required")
    grouped = {str(well_id): rows.sort_values("row_index").copy() for well_id, rows in matrix.groupby("well_id", sort=False)}
    data_dir = args.data_dir.resolve()
    jobs = int(config.get("runtime", {}).get("n_jobs", 1))
    results = Parallel(n_jobs=jobs, prefer="threads", verbose=10)(
        delayed(_process_well)(well_id, grouped[well_id], data_dir, branches) for well_id in well_ids.astype(str)
    )
    reports = []
    for well_id, corrections, diagnostics in results:
        indices = grouped[well_id].index
        for name, correction in corrections.items():
            matrix.loc[indices, f"correction_{name}"] = correction
        reports.extend(diagnostics)
    selection = dict(config.get("selection", {}))
    specs = [(branch, float(weight)) for branch in branches for weight in selection.get("weights", [0.1, 0.2, 0.35, 0.5])]
    nested, selections, inference, spec_report = _nested_specs(
        matrix, specs, float(selection.get("minimum_selection_gain", 0.02)),
        float(selection.get("inner_fold_tolerance", 0.0)),
    )
    baseline = _evaluation_frame(matrix, matrix["base_y_pred"].to_numpy(dtype=float))
    candidate = _evaluation_frame(matrix, nested)
    validation = dict(config.get("validation", {}))
    gate = evaluate_candidate_gates(
        baseline, candidate,
        minimum_standard_gain=float(validation.get("minimum_standard_gain", 0.05)),
        minimum_improved_fold_fraction=float(validation.get("minimum_improved_fold_fraction", 0.8)),
        bootstrap_resamples=int(validation.get("bootstrap_resamples", 2000)), seed=int(config.get("seed", 42)),
    )
    gate["inference_spec"] = inference
    gate["promoted_to_spatial_audit"] = bool(gate["promoted"] and inference is not None)
    gate["promoted"] = False
    gate["promotion_note"] = "Stage 10 requires a separate spatial audit before Kaggle integration."
    gate["selections"] = selections
    gate["spec_report"] = spec_report
    matrix["y_pred"] = nested.astype(np.float32)
    matrix.to_parquet(output_dir / "oof.parquet", index=False)
    pd.DataFrame(reports).to_parquet(output_dir / "well_diagnostics.parquet", index=False)
    write_json(output_dir / "gate_summary.json", gate)
    write_json(output_dir / "environment.json", environment_report())
    resolved = dict(config)
    resolved["resolved"] = {"cutback_run": str(args.cutback_run.resolve()), "data_dir": str(data_dir), "run_id": args.run_id, "limit_wells": args.limit_wells}
    write_yaml(output_dir / "config.yaml", resolved)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"run artifacts: {output_dir}")


if __name__ == "__main__":
    main()
