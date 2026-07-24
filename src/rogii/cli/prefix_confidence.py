from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.branch_retrieval import _bootstrap, _metrics
from rogii.cli.prefix_router import (
    FOLD_FAMILIES,
    _family_report,
    _guarded_route,
    build_candidates,
    internal_cut_indices,
)
from rogii.cli.selector_resolution import stable_stratified_sample
from rogii.cli.trajectory_residual import _typewell_folds
from rogii.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 21B disjoint visible-prefix confidence gate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--public-oof-run", type=Path, required=True)
    parser.add_argument("--stage21a-run", type=Path, required=True)
    parser.add_argument("--exclude-run", type=Path, action="append", default=[])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit-cuts", type=int)
    return parser


def calibration_penalties(
    frame: pd.DataFrame,
    candidate_names: list[str],
    mad_weight: float,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    """Estimate candidate-specific inner-to-outer optimism on calibration wells."""
    penalties: dict[str, float] = {}
    rows: list[dict[str, float | int | str]] = []
    for name in candidate_names:
        values = (
            frame.loc[frame["candidate"].astype(str) == name, "outer_rmse"].to_numpy(float)
            - frame.loc[frame["candidate"].astype(str) == name, "inner_rmse"].to_numpy(float)
        )
        if len(values) < 10 or not np.isfinite(values).all():
            raise ValueError(f"{name}: insufficient finite Stage 21A calibration rows")
        median = float(np.median(values))
        mad = float(1.4826 * np.median(np.abs(values - median)))
        penalty = median + float(mad_weight) * mad
        penalties[name] = penalty
        rows.append(
            {
                "candidate": name,
                "cuts": int(len(values)),
                "median_optimism": median,
                "optimism_mad": mad,
                "risk_adjusted_penalty": penalty,
            }
        )
    return penalties, rows


def confidence_choice(
    inner_scores: dict[str, list[float]],
    penalties: dict[str, float],
    base_name: str,
    minimum_corrected_margin: float,
    minimum_inner_margin: float,
) -> tuple[str, str, dict[str, float]]:
    aggregate = {name: float(np.median(values)) for name, values in inner_scores.items()}
    corrected = {name: aggregate[name] + float(penalties[name]) for name in aggregate}
    proposed = min(corrected, key=lambda name: (corrected[name], name))
    if proposed == base_name:
        return base_name, "base_ranked_first", corrected
    if corrected[proposed] > corrected[base_name] - float(minimum_corrected_margin):
        return base_name, "corrected_margin", corrected
    agreement = all(
        selected + float(minimum_inner_margin) <= base
        for selected, base in zip(inner_scores[proposed], inner_scores[base_name], strict=True)
    )
    if not agreement:
        return base_name, "inner_disagreement", corrected
    return proposed, "alternative_accepted", corrected


def _weighted_route(
    base: np.ndarray,
    selected: np.ndarray,
    router_config: dict[str, Any],
    weight: float,
) -> np.ndarray:
    return _guarded_route(base, selected, {**router_config, "blend_weight": float(weight)})


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    stage16 = args.stage16b_run.resolve()
    stage17 = args.stage17a_run.resolve()
    public_run = args.public_oof_run.resolve()
    stage21a = args.stage21a_run.resolve()
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    summary17 = json.loads((stage17 / "summary.json").read_text(encoding="utf-8"))
    summary21a = json.loads((stage21a / "summary.json").read_text(encoding="utf-8"))
    if summary17.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 17A manifest provenance mismatch")
    if summary21a.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 21A manifest provenance mismatch")
    if summary21a.get("promoted_to_stage21b") is not False:
        raise AssertionError("Stage 21B is specifically the repair path for rejected Stage 21A")

    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    candidate_config = dict(config.get("candidates", {}))
    candidate_names = ["public_oof"]
    for multiplier in [float(value) for value in candidate_config.get("gr_sigma_multipliers", [])]:
        tag = f"a{int(round(multiplier * 100)):03d}"
        candidate_names.extend([f"selector_{tag}", f"top_pf_{tag}"])
    if any(name.startswith("poly_") for name in candidate_names):
        raise AssertionError("Stage 21B must not include polynomial candidates")
    calibration_frame = pd.read_parquet(stage21a / "candidate_report.parquet")
    calibration_config = dict(config.get("calibration", {}))
    penalties, penalty_rows = calibration_penalties(
        calibration_frame,
        candidate_names,
        float(calibration_config.get("residual_mad_weight", 0.5)),
    )

    excluded_wells = set(
        pd.read_parquet(stage21a / "router_cut_report.parquet", columns=["well_id"])["well_id"].astype(str)
    )
    for run in args.exclude_run:
        path = run.resolve() / "cut_features.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        excluded_wells.update(pd.read_parquet(path, columns=["well_id"])["well_id"].astype(str))

    cuts = pd.read_parquet(stage17 / "cut_report.parquet")
    eligible = cuts[(cuts["evaluation_role"] == "primary") & cuts["replay_eligible"]].copy()
    eligible = eligible[
        eligible["cut_index"].astype(int) - eligible["original_public_cut_index"].astype(int)
        >= int(calibration_config.get("minimum_calibration_gap_rows", 96))
    ].copy()
    eligible = eligible[~eligible["well_id"].astype(str).isin(excluded_wells)].copy()
    selected = stable_stratified_sample(eligible, int(config.get("sample", {}).get("cuts_per_stratum", 4)))
    if args.limit_cuts is not None:
        selected = selected.head(int(args.limit_cuts)).copy()
    selected = selected.sort_values(["well_id", "cut_index"], kind="stable").reset_index(drop=True)
    overlap = sorted(set(selected["well_id"].astype(str)).intersection(excluded_wells))
    if overlap:
        raise AssertionError(f"Calibration/discovery well leakage: {overlap[:5]}")

    assignments = pd.read_parquet(stage16 / "well_assignments.parquet")
    assignments["well_id"] = assignments["well_id"].astype(str)
    assignments["typewell_fold"] = _typewell_folds(
        assignments, int(config.get("validation", {}).get("n_typewell_folds", 5)), seed
    )
    selected = selected.merge(
        assignments[["well_id", "spatial_fold", "typewell_fold", "branch_group_fold"]],
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    selected["stage16_fold"] = selected["stage16_fold"].astype(int)
    selected_wells = selected["well_id"].astype(str).unique().tolist()
    public = pd.read_parquet(
        public_run / "base_oof.parquet",
        columns=["well_id", "row_index", "y_pred"],
        filters=[("well_id", "in", selected_wells)],
    )
    public["well_id"] = public["well_id"].astype(str)
    public_by_well = {
        well: frame.sort_values("row_index") for well, frame in public.groupby("well_id", sort=True)
    }
    train = args.data_dir.resolve() / "train"

    @lru_cache(maxsize=64)
    def load_well(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train / f"{well_id}__horizontal_well.csv")

    @lru_cache(maxsize=64)
    def load_typewell(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train / f"{well_id}__typewell.csv")

    router_config = dict(config.get("router", {}))
    base_name = str(router_config.get("base_candidate", "top_pf_a130"))
    diagnostic_weights = [float(value) for value in config.get("diagnostic_weights", [0.1])]
    primary_weight = float(router_config.get("blend_weight", 0.1))
    if primary_weight not in diagnostic_weights:
        diagnostic_weights.append(primary_weight)
    cut_rows: list[dict[str, Any]] = []
    invariance: list[bool] = []
    for position, cut in enumerate(selected.itertuples(index=False), 1):
        well_id, outer = str(cut.well_id), int(cut.cut_index)
        horizontal, typewell = load_well(well_id), load_typewell(well_id)
        source = public_by_well[well_id]
        original = int(source["row_index"].min())
        if original != int(cut.original_public_cut_index):
            raise AssertionError(f"{well_id}: public OOF cutoff mismatch")
        source_prediction = source["y_pred"].to_numpy(float)
        inner_cuts = internal_cut_indices(original, outer, calibration_config)
        if len(inner_cuts) < 2:
            raise AssertionError(f"{cut.cut_id}: fewer than two internal calibration cuts")
        inner_scores: dict[str, list[float]] = {}
        for inner in inner_cuts:
            candidates = build_candidates(
                horizontal, typewell, inner, source_prediction[inner - original :], candidate_config
            )
            truth = horizontal["TVT"].to_numpy(float)[inner:outer]
            holdout = outer - inner
            for name, prediction in candidates.items():
                inner_scores.setdefault(name, []).append(
                    float(np.sqrt(np.mean(np.square(prediction[:holdout] - truth))))
                )
        selected_name, decision, corrected = confidence_choice(
            inner_scores,
            penalties,
            base_name,
            float(router_config.get("minimum_corrected_margin", 0.5)),
            float(router_config.get("minimum_inner_margin", 0.0)),
        )
        outer_candidates = build_candidates(
            horizontal, typewell, outer, source_prediction[outer - original :], candidate_config
        )
        base, alternative = outer_candidates[base_name], outer_candidates[selected_name]
        truth = horizontal["TVT"].to_numpy(float)[outer:]
        row: dict[str, Any] = {
            "cut_id": str(cut.cut_id),
            "well_id": well_id,
            "requested_fraction": float(cut.requested_fraction),
            **{family: int(getattr(cut, family)) for family in FOLD_FAMILIES},
            "suffix_rows": len(truth),
            "base_sse": float(np.square(base - truth).sum()),
            "selected_candidate": selected_name,
            "decision": decision,
            "corrected_margin": float(corrected[base_name] - corrected[selected_name]),
        }
        for weight in diagnostic_weights:
            prediction = _weighted_route(base, alternative, router_config, weight)
            row[f"candidate_sse_w{int(round(weight * 1000)):03d}"] = float(
                np.square(prediction - truth).sum()
            )
        cut_rows.append(row)
        if len(invariance) < 8:
            changed = horizontal.copy()
            changed.loc[changed.index >= outer, "TVT"] += 9999.0
            changed_candidates = build_candidates(
                changed, typewell, outer, source_prediction[outer - original :], candidate_config
            )
            invariance.append(
                all(np.array_equal(outer_candidates[name], changed_candidates[name]) for name in outer_candidates)
            )
        if position % 10 == 0:
            print(f"prefix confidence {position}/{len(selected)} cuts", flush=True)

    report = pd.DataFrame.from_records(cut_rows)
    primary_column = f"candidate_sse_w{int(round(primary_weight * 1000)):03d}"
    report["candidate_sse"] = report[primary_column]
    metrics = _metrics(report)
    bootstrap = _bootstrap(
        report, int(config.get("validation", {}).get("bootstrap_resamples", 2000)), seed
    )
    family_reports = {family: _family_report(report, family) for family in FOLD_FAMILIES}
    base_well = report.groupby("well_id").agg(sse=("base_sse", "sum"), rows=("suffix_rows", "sum"))
    candidate_well = report.groupby("well_id").agg(
        sse=("candidate_sse", "sum"), rows=("suffix_rows", "sum")
    )
    p90_delta = float(
        np.sqrt(candidate_well.sse / candidate_well.rows).quantile(0.9)
        - np.sqrt(base_well.sse / base_well.rows).quantile(0.9)
    )
    weight_report = []
    rows = int(report["suffix_rows"].sum())
    base_rmse = float(np.sqrt(report["base_sse"].sum() / rows))
    for weight in sorted(diagnostic_weights):
        column = f"candidate_sse_w{int(round(weight * 1000)):03d}"
        candidate_rmse = float(np.sqrt(report[column].sum() / rows))
        weight_report.append(
            {
                "weight": weight,
                "base_rmse": base_rmse,
                "candidate_rmse": candidate_rmse,
                "rmse_delta": candidate_rmse - base_rmse,
            }
        )
    validation = dict(config.get("validation", {}))
    gates = {
        "hidden_target_invariance": bool(invariance) and all(invariance),
        "public_oof_target_safe": True,
        "calibration_well_overlap_zero": len(overlap) == 0,
        "primary_gain": metrics["rmse_delta"] <= -float(validation.get("minimum_gain", 0.05)),
        "bootstrap_upper_below_zero": bootstrap[1] < 0,
        "standard_fold_consistency": metrics["improved_folds"]
        >= int(validation.get("minimum_improved_folds", 4)),
        "fraction_consistency": metrics["improved_fractions"]
        >= int(validation.get("minimum_improved_fractions", 4)),
        "spatial_fold_consistency": family_reports["spatial_fold"]["improved_folds"]
        >= int(validation.get("minimum_spatial_folds", 4)),
        "typewell_fold_consistency": family_reports["typewell_fold"]["improved_folds"]
        >= int(validation.get("minimum_typewell_folds", 4)),
        "branch_group_fold_consistency": family_reports["branch_group_fold"]["improved_folds"]
        >= int(validation.get("minimum_branch_folds", 4)),
        "well_p90_nonworse": p90_delta <= 0,
    }
    promoted = bool(all(gates.values()))
    report.to_parquet(output / "confidence_cut_report.parquet", index=False)
    pd.DataFrame.from_records(penalty_rows).to_parquet(
        output / "calibration_penalties.parquet", index=False
    )
    summary = {
        "stage21b_complete": True,
        "promoted_to_stage21c": promoted,
        "stage16b_manifest_sha256": expected_hash,
        "calibration_cuts": int(calibration_frame["cut_id"].nunique()),
        "calibration_wells": int(calibration_frame["well_id"].nunique()),
        "sample_cuts": len(report),
        "sample_wells": int(report["well_id"].nunique()),
        "excluded_wells": len(excluded_wells),
        "calibration_well_overlap": overlap,
        "candidate_count": len(candidate_names),
        "alternative_accepted_cuts": int((report["decision"] == "alternative_accepted").sum()),
        "selected_candidate_counts": {
            str(key): int(value) for key, value in report["selected_candidate"].value_counts().items()
        },
        "base_rmse": metrics["base_rmse"],
        "candidate_rmse": metrics["candidate_rmse"],
        "rmse_delta": metrics["rmse_delta"],
        "well_p90_delta": p90_delta,
        "bootstrap_95pct": bootstrap,
        "metrics": metrics,
        "family_reports": family_reports,
        "weight_report": weight_report,
        "calibration_penalties": penalty_rows,
        "gates": gates,
        "next_step": (
            "Run high-resolution all-cut confirmation before any Kaggle integration."
            if promoted
            else "Stop prefix candidate routing and move to a different alignment target/model family."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16),
        "stage17a_run": str(stage17),
        "public_oof_run": str(public_run),
        "stage21a_run": str(stage21a),
        "exclude_runs": [str(path.resolve()) for path in args.exclude_run],
        "data_dir": str(args.data_dir.resolve()),
        "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
