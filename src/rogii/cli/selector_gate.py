from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config


FEATURES = [
    "requested_fraction", "cut_index", "suffix_rows", "prefix_fraction",
    "selector_code", "scale", "configured_beam_weight", "hold_weight",
    "tracking_steps", "tracking_fraction", "gr_sigma", "likelihood_spread",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-fit a target-free gate for Stage 17B selector cuts")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage17b-run", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _model(config: dict[str, Any], seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=float(config.get("learning_rate", 0.05)),
        max_iter=int(config.get("max_iter", 180)),
        max_leaf_nodes=int(config.get("max_leaf_nodes", 15)),
        min_samples_leaf=int(config.get("min_samples_leaf", 25)),
        l2_regularization=float(config.get("l2_regularization", 15.0)),
        random_state=seed,
    )


def _feature_frame(cuts: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    keys = ["cut_id", "well_id"]
    duplicate_columns = [column for column in audit.columns if column in cuts.columns and column not in keys]
    audit_features = audit.drop(columns=duplicate_columns)
    uncovered = cuts[~cuts["replay_eligible"]].merge(
        audit_features, on=keys, validate="one_to_one"
    )
    uncovered["prefix_fraction"] = uncovered["cut_index"] / (uncovered["cut_index"] + uncovered["suffix_rows"])
    uncovered["tracking_fraction"] = uncovered["tracking_steps"] / uncovered["suffix_rows"]
    uncovered["selector_rmse"] = np.sqrt(uncovered["selector_sse"] / uncovered["suffix_rows"])
    uncovered["baseline_rmse"] = np.sqrt(uncovered["baseline_sse"] / uncovered["suffix_rows"])
    uncovered["gain_target"] = uncovered["baseline_rmse"] - uncovered["selector_rmse"]
    for feature in FEATURES:
        uncovered[feature] = pd.to_numeric(uncovered[feature], errors="coerce")
        uncovered[feature] = uncovered[feature].fillna(float(uncovered[feature].median()))
    return uncovered


def crossfit_gate(frame: pd.DataFrame, config: dict[str, Any], seed: int) -> np.ndarray:
    output = np.full(len(frame), np.nan, dtype=float)
    folds = frame["stage16_fold"].to_numpy(dtype=np.int16)
    for fold in sorted(np.unique(folds)):
        valid = folds == fold
        model = _model(config, seed + int(fold))
        model.fit(
            frame.loc[~valid, FEATURES].to_numpy(np.float32),
            frame.loc[~valid, "gain_target"].to_numpy(float),
            sample_weight=frame.loc[~valid, "suffix_rows"].to_numpy(float),
        )
        output[valid] = model.predict(frame.loc[valid, FEATURES].to_numpy(np.float32))
    if not np.isfinite(output).all():
        raise AssertionError("Gate cross-fit produced incomplete predictions")
    return output


def _rmse(sse: float, rows: int) -> float:
    return float(np.sqrt(float(sse) / int(rows))) if rows else float("nan")


def _profile(cuts: pd.DataFrame, gate: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    decisions = gate.set_index("cut_id")["predicted_gain"].ge(float(threshold))
    result = cuts.copy()
    use = result["cut_id"].map(decisions).fillna(False).to_numpy(bool)
    result["gate_selected"] = use
    result["gated_sse"] = np.where(use, result["selector_sse"], result["hybrid_sse"])
    primary = result[result["evaluation_role"] == "primary"]
    rows = int(primary["suffix_rows"].sum())
    always_rmse = _rmse(float(primary["candidate_sse"].sum()), rows)
    gated_rmse = _rmse(float(primary["gated_sse"].sum()), rows)
    fold_deltas: dict[str, float] = {}
    for fold, part in primary.groupby("stage16_fold", sort=True):
        fold_rows = int(part["suffix_rows"].sum())
        fold_deltas[f"fold_{int(fold)}"] = _rmse(float(part["gated_sse"].sum()), fold_rows) - _rmse(float(part["candidate_sse"].sum()), fold_rows)
    cut_always = np.sqrt(primary["candidate_sse"] / primary["suffix_rows"])
    cut_gated = np.sqrt(primary["gated_sse"] / primary["suffix_rows"])
    metrics = {
        "threshold": float(threshold), "selected_cuts": int(primary["gate_selected"].sum()),
        "selected_fraction": float(primary["gate_selected"].mean()),
        "always_selector_rmse": always_rmse, "gated_rmse": gated_rmse,
        "rmse_delta": gated_rmse - always_rmse,
        "cut_rmse_p90_delta": float(cut_gated.quantile(0.9) - cut_always.quantile(0.9)),
        "cut_rmse_max_delta": float(cut_gated.max() - cut_always.max()),
        "fold_deltas": fold_deltas,
    }
    return result, metrics


def _well_bootstrap(candidate: pd.DataFrame, baseline: pd.DataFrame, resamples: int, seed: int) -> list[float]:
    def aggregate(frame: pd.DataFrame, column: str) -> pd.Series:
        grouped = frame.groupby("well_id", sort=True).agg(sse=(column, "sum"), rows=("suffix_rows", "sum"))
        return np.sqrt(grouped["sse"] / grouped["rows"])
    left, right = aggregate(candidate, "gated_sse"), aggregate(baseline, "candidate_sse")
    delta = (left - right).to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(delta, size=(resamples, len(delta)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    source = args.stage17b_run.resolve()
    cuts = pd.read_parquet(source / "cut_report.parquet")
    audit = pd.read_parquet(source / "selector_audit.parquet")
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    if source_summary.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 17B did not use the frozen manifest")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_config = dict(config.get("model", {}))
    seed = int(config.get("seed", 42))
    gate = _feature_frame(cuts, audit)
    gate["predicted_gain"] = crossfit_gate(gate, model_config, seed)
    gate["prediction_error"] = gate["predicted_gain"] - gate["gain_target"]
    threshold = float(config.get("primary_threshold", 0.0))
    gated_cuts, primary_metrics = _profile(cuts, gate, threshold)
    grid = []
    for value in config.get("diagnostic_thresholds", [-2.0, 0.0, 2.0, 4.0]):
        _, metrics = _profile(cuts, gate, float(value)); grid.append(metrics)
    primary = gated_cuts[gated_cuts["evaluation_role"] == "primary"]
    ci = _well_bootstrap(primary, primary, int(config.get("bootstrap_resamples", 1000)), seed)
    fold_deltas = list(primary_metrics["fold_deltas"].values())
    gates_config = dict(config.get("gates", {}))
    gates = {
        "crossfit_well_isolated": True,
        "pooled_gain": primary_metrics["rmse_delta"] <= -float(gates_config.get("minimum_gain", 0.02)),
        "fold_consistency": sum(value < 0 for value in fold_deltas) >= int(gates_config.get("minimum_improved_folds", 4)),
        "worst_fold_nonworse": max(fold_deltas) <= float(gates_config.get("maximum_worst_fold_delta", 0.02)),
        "p90_nonworse": primary_metrics["cut_rmse_p90_delta"] <= float(gates_config.get("maximum_p90_delta", 0.0)),
        "bootstrap_upper_below_zero": ci[1] < 0,
    }
    full_model = _model(model_config, seed)
    full_model.fit(
        gate[FEATURES].to_numpy(np.float32), gate["gain_target"].to_numpy(float),
        sample_weight=gate["suffix_rows"].to_numpy(float),
    )
    with (output / "selector_gate.pkl").open("wb") as handle:
        pickle.dump(full_model, handle)
    gate.to_parquet(output / "gate_oof.parquet", index=False)
    gated_cuts.to_parquet(output / "cut_report.parquet", index=False)
    write_json(output / "feature_columns.json", FEATURES)
    summary = {
        "stage17c_complete": True, "promoted_to_resolution_audit": bool(all(gates.values())),
        "stage16b_manifest_sha256": expected_hash, "primary_threshold": threshold,
        "primary_metrics": primary_metrics, "bootstrap_95pct": ci,
        "gain_prediction_correlation": float(np.corrcoef(gate["predicted_gain"], gate["gain_target"])[0, 1]),
        "threshold_grid": grid, "gates": gates,
        "next_step": "Run a stratified medium/full-resolution PF audit if the target-free gate promotes.",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {"stage17b_run": str(source), "run_id": args.run_id}
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
