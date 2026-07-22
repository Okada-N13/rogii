from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.evaluation.delta_u_gate import absolute_tail_metrics, nested_select_predictions, prediction_report
from rogii.evaluation.metrics import evaluate_predictions, paired_well_bootstrap
from rogii.models.emission_residual import GENERIC_FEATURES, STACKED_FEATURES, generic_residual_features, make_residual_model, stacked_residual_features, target_invariance_check


FAMILIES = ("fold", "spatial_fold", "typewell_fold")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 14 cross-fitted emission residual model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage12b-run", type=Path, required=True)
    parser.add_argument("--stage12c-run", type=Path, required=True)
    parser.add_argument("--stage13-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--limit-wells", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def _load_family(root: Path, family: str, limit_wells: int | None) -> pd.DataFrame:
    frame = pd.read_parquet(root / f"{family}_path_oof.parquet")
    if limit_wells is not None:
        wells = frame["well_id"].drop_duplicates().iloc[:limit_wells]
        frame = frame[frame["well_id"].isin(set(wells))].copy()
    required = {"id", "well_id", "cut_id", "cut_fraction", "row_index", "MD", "y_true", "surface_y_pred", "y_pred", "fold"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{family} is missing Stage 14 columns: {missing}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{family} contains duplicate IDs")
    return frame.sort_values("id").reset_index(drop=True)


def _crossfit_raw_residual(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    family: str,
    branch: str,
    model_config: dict[str, Any],
    seed: int,
    output: Path,
    resume: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    folds = frame["fold"].to_numpy(int)
    target = frame["y_true"].to_numpy(float) - frame["y_pred"].to_numpy(float)
    raw = np.zeros(len(frame), np.float32)
    reports = []
    maximum_rows = int(model_config.get("maximum_training_rows", 300_000))
    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        valid_mask = ~train_mask
        train_wells = set(frame.loc[train_mask, "well_id"].astype(str))
        valid_wells = set(frame.loc[valid_mask, "well_id"].astype(str))
        if train_wells & valid_wells:
            raise AssertionError(f"{family}/{branch}/fold{fold}: well leakage")
        train_index = np.flatnonzero(train_mask)
        if maximum_rows > 0 and len(train_index) > maximum_rows:
            rng = np.random.default_rng(seed + int(fold) * 1009 + len(branch))
            train_index = np.sort(rng.choice(train_index, maximum_rows, replace=False))
        model_path = output / f"{family}_{branch}_fold_{fold}.joblib"
        if resume and model_path.is_file():
            model = joblib.load(model_path)
            reused = True
        else:
            model = make_residual_model(model_config, seed + int(fold) * 17 + len(branch) * 101)
            model.fit(features.iloc[train_index].to_numpy(np.float32), target[train_index])
            joblib.dump(model, model_path, compress=3)
            reused = False
        raw[valid_mask] = model.predict(features.loc[valid_mask].to_numpy(np.float32)).astype(np.float32)
        reports.append(
            {
                "family": family,
                "branch": branch,
                "fold": int(fold),
                "training_rows": int(len(train_index)),
                "validation_rows": int(valid_mask.sum()),
                "training_wells": int(len(train_wells)),
                "validation_wells": int(len(valid_wells)),
                "checkpoint_reused": reused,
            }
        )
        print(reports[-1], flush=True)
    return raw, reports


def _candidate_specs(
    base: np.ndarray,
    corrections: dict[str, np.ndarray],
    specs: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for branch, raw in corrections.items():
        for spec in specs:
            name = f"{branch}_{spec['name']}"
            cap = float(spec.get("cap", 1e9))
            weight = float(spec.get("weight", 1.0))
            output[name] = base + weight * np.clip(raw, -cap, cap)
    return output


def _family_report(base: pd.DataFrame, predictions: dict[str, np.ndarray], selection: dict[str, Any]):
    nested, selections = nested_select_predictions(base, predictions, selection)
    base_metrics, _ = evaluate_predictions(base)
    nested_metrics, _ = evaluate_predictions(nested)
    reports = {name: prediction_report(base, values) for name, values in predictions.items()}
    bootstrap = paired_well_bootstrap(nested, base, n_resamples=int(selection.get("bootstrap_resamples", 2000)), seed=42)
    return nested, {
        "base_metrics": base_metrics,
        "nested_metrics": nested_metrics,
        "nested_rmse_delta": float(nested_metrics["pooled_rmse"] - base_metrics["pooled_rmse"]),
        "base_tail": absolute_tail_metrics(base),
        "nested_tail": absolute_tail_metrics(nested),
        "bootstrap": bootstrap,
        "selections": selections,
        "profile_reports": reports,
    }


def _robust_generic_spec(family_reports: dict[str, Any], specs: list[dict[str, Any]], tolerance: float) -> str | None:
    eligible: list[tuple[float, str]] = []
    for spec in specs:
        name = f"generic_{spec['name']}"
        reports = [family_reports[family]["profile_reports"][name] for family in FAMILIES]
        if all(
            report["pooled_rmse_delta"] < 0.0
            and max(report["fold_deltas"].values(), default=0.0) <= tolerance
            for report in reports
        ):
            eligible.append((sum(float(report["pooled_rmse_delta"]) for report in reports), name))
    return min(eligible)[1] if eligible else None


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    model_config = dict(config.get("model", {}))
    specs = [dict(value) for value in config.get("correction_specs", [])]
    selection = dict(config.get("selection", {}))
    validation = dict(config.get("validation", {}))
    if not specs or len({row["name"] for row in specs}) != len(specs):
        raise ValueError("Stage 14 correction specs must be present and unique")
    stage12b, stage12c, stage13 = args.stage12b_run.resolve(), args.stage12c_run.resolve(), args.stage13_run.resolve()
    summary12b = load_config(stage12b / "gate_summary.json")
    summary13 = load_config(stage13 / "gate_summary.json")
    if not summary12b.get("promoted_to_spatial_emission_audit"):
        raise RuntimeError("Stage 12B did not authorize residual modeling")
    if summary13.get("promoted_to_all_train_uncertainty_model"):
        raise RuntimeError("Stage 13 already promoted; Stage 14 fallback is unnecessary")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    frames = {family: _load_family(stage12c, family, args.limit_wells) for family in FAMILIES}
    ids = frames["fold"]["id"].to_numpy(str)
    alignment = {}
    for family, frame in frames.items():
        alignment[family] = {
            "rows": int(len(frame)),
            "id_order_matches": bool(np.array_equal(frame["id"].to_numpy(str), ids)),
            "target_max_abs_difference": float(np.max(np.abs(frame["y_true"].to_numpy(float) - frames["fold"]["y_true"].to_numpy(float)))),
        }
    if not all(row["id_order_matches"] and row["target_max_abs_difference"] == 0.0 for row in alignment.values()):
        raise AssertionError(f"Stage 14 alignment failed: {alignment}")
    entropy_frame = pd.read_parquet(stage12b / "oof_emission.parquet", columns=["id", "learned_entropy"])
    entropy = entropy_frame.set_index("id").reindex(ids)["learned_entropy"]
    if entropy.isna().any():
        raise AssertionError("Stage 14 entropy alignment failed")

    invariance = {family: target_invariance_check(frame.head(min(len(frame), 20_000))) for family, frame in frames.items()}
    if not all(invariance.values()):
        raise AssertionError(f"Stage 14 hidden-target invariance failed: {invariance}")
    family_reports, training_reports = {}, []
    for family, frame in frames.items():
        generic = generic_residual_features(frame)
        raw_generic, report = _crossfit_raw_residual(
            frame, generic, family, "generic", model_config, int(config.get("seed", 42)), output, args.resume
        )
        training_reports.extend(report)
        corrections = {"generic": raw_generic}
        if family == "fold":
            stacked = stacked_residual_features(
                frame,
                frames["spatial_fold"]["y_pred"].to_numpy(float),
                frames["typewell_fold"]["y_pred"].to_numpy(float),
                entropy.to_numpy(float),
            )
            raw_stacked, report = _crossfit_raw_residual(
                frame, stacked, family, "stacked", model_config, int(config.get("seed", 42)) + 1000, output, args.resume
            )
            training_reports.extend(report)
            corrections["stacked"] = raw_stacked
        predictions = _candidate_specs(frame["y_pred"].to_numpy(float), corrections, specs)
        if family == "fold":
            predictions["control_blend_s90_sp05_tw05"] = (
                0.90 * frame["y_pred"].to_numpy(float)
                + 0.05 * frames["spatial_fold"]["y_pred"].to_numpy(float)
                + 0.05 * frames["typewell_fold"]["y_pred"].to_numpy(float)
            )
        nested, family_report = _family_report(frame, predictions, selection)
        family_reports[family] = family_report
        artifact = frame[["id", "well_id", "cut_id", "cut_fraction", "row_index", "MD", "y_true", "surface_y_pred", "y_pred", "fold"]].copy()
        artifact["raw_generic_residual"] = raw_generic
        if "stacked" in corrections:
            artifact["raw_stacked_residual"] = corrections["stacked"]
        artifact["nested_y_pred"] = nested["y_pred"].to_numpy(float)
        artifact.to_parquet(output / f"{family}_residual_oof.parquet", index=False)

    robust_name = _robust_generic_spec(
        family_reports, specs, float(validation.get("inference_fold_tolerance", 0.03))
    )
    standard, spatial, typewell = (family_reports[name] for name in FAMILIES)
    gates = {
        "hidden_target_invariance": all(invariance.values()),
        "standard_nested_gain": standard["nested_rmse_delta"] <= -float(validation.get("minimum_standard_gain", 0.05)),
        "standard_bootstrap": standard["bootstrap"]["ci_97_5"] < 0.0,
        "standard_p90_nonworse": standard["nested_metrics"]["well_rmse_p90"] <= standard["base_metrics"]["well_rmse_p90"] * (1.0 + float(validation.get("tail_tolerance", 0.005))),
        "standard_worst10_nonworse": standard["nested_metrics"]["worst_10pct_sse_share"] <= standard["base_metrics"]["worst_10pct_sse_share"] * (1.0 + float(validation.get("tail_tolerance", 0.005))),
        "spatial_nested_gain": spatial["nested_rmse_delta"] <= -float(validation.get("minimum_holdout_gain", 0.02)),
        "typewell_nested_gain": typewell["nested_rmse_delta"] <= -float(validation.get("minimum_holdout_gain", 0.02)),
        "robust_generic_spec": robust_name is not None,
    }
    promoted = all(gates.values())
    summary = {
        "promoted_to_full_residual_training": promoted,
        "experiment": "stage14_crossfit_emission_residual",
        "alignment": alignment,
        "feature_columns": {"generic": list(GENERIC_FEATURES), "stacked": list(STACKED_FEATURES)},
        "hidden_target_invariance": invariance,
        "family_reports": family_reports,
        "training_reports": training_reports,
        "robust_generic_spec": robust_name,
        "gates": gates,
        "next_step": "Train full-data generic/stacked residual ensemble and build independent test inference." if promoted else "Keep the 90/5/5 control and redesign the emission model rather than adding more postprocessing.",
    }
    write_json(output / "gate_summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {"stage12b_run": str(stage12b), "stage12c_run": str(stage12c), "stage13_run": str(stage13), "artifact_dir": str(args.artifact_dir.resolve()), "run_id": args.run_id, "limit_wells": args.limit_wells, "resume": args.resume}
    write_yaml(output / "config.yaml", config)
    print({"promoted_to_full_residual_training": promoted, "standard_delta": standard["nested_rmse_delta"], "spatial_delta": spatial["nested_rmse_delta"], "typewell_delta": typewell["nested_rmse_delta"], "bootstrap_95pct": {family: [report["bootstrap"]["ci_2_5"], report["bootstrap"]["ci_97_5"]] for family, report in family_reports.items()}, "robust_generic_spec": robust_name, "gates": gates, "next_step": summary["next_step"]}, flush=True)
    print(f"run artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
