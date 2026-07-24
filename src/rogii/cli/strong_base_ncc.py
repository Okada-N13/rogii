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
from rogii.cli.prefix_router import FOLD_FAMILIES, _family_report, build_candidates
from rogii.cli.trajectory_residual import _typewell_folds
from rogii.config import load_config
from rogii.models.emission_lattice import decode_lattice
from rogii.models.raw_ncc import offset_grid, rolling_ncc_cost


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 23A strong-base physical offset-state audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--public-oof-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit-cuts", type=int)
    return parser


def strong_base_costs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cut_index: int,
    base: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    stride = int(config.get("alignment_stride", 4))
    if stride < 1:
        raise ValueError("alignment_stride must be positive")
    full_positions = np.arange(int(cut_index), len(horizontal), dtype=np.int64)
    take = np.arange(0, len(full_positions), stride, dtype=np.int64)
    maximum = int(config.get("max_eval_rows_per_cut", 512))
    if maximum > 0 and len(take) > maximum:
        take = take[np.unique(np.linspace(0, len(take) - 1, maximum).round().astype(int))]
    positions = full_positions[take]
    surface = np.asarray(base, float)[take]
    observed_gr = pd.to_numeric(horizontal.iloc[positions]["GR"], errors="coerce").to_numpy(float)
    offsets = offset_grid(config)
    candidate_tvt = surface[:, None] + offsets[None, :]
    type_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(float)
    type_gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(float)
    finite = np.isfinite(type_tvt) & np.isfinite(type_gr)
    type_tvt, type_gr = type_tvt[finite], type_gr[finite]
    order = np.argsort(type_tvt)
    type_tvt, type_gr = type_tvt[order], type_gr[order]
    if len(type_tvt) < 5:
        raise ValueError("Insufficient finite typewell GR")
    expected = np.interp(candidate_tvt.ravel(), type_tvt, type_gr).reshape(candidate_tvt.shape)
    valid_expected = (candidate_tvt >= type_tvt[0]) & (candidate_tvt <= type_tvt[-1])
    costs: dict[str, np.ndarray] = {}
    for window in [int(value) for value in config.get("windows", [5, 13, 25])]:
        costs[f"ncc_w{window}"] = rolling_ncc_cost(
            observed_gr, expected, valid_expected, window
        )
    mix_windows = [int(value) for value in config.get("mix_windows", [13, 25])]
    mix_weights = np.asarray(config.get("mix_weights", [0.4, 0.6]), float)
    if len(mix_windows) != len(mix_weights) or not np.isclose(mix_weights.sum(), 1.0):
        raise ValueError("Invalid NCC mix")
    mix = np.zeros_like(next(iter(costs.values())))
    for window, weight in zip(mix_windows, mix_weights, strict=True):
        mix += float(weight) * costs[f"ncc_w{window}"]
    costs["ncc_mix"] = mix
    return positions, surface, offsets, costs


def _signal_counts(frame: pd.DataFrame, column: str, random_top10: float) -> tuple[int, dict[str, float]]:
    values = {}
    for key, group in frame.groupby(column, sort=True):
        valid = group.loc[group["emission_valid"], "top10"]
        values[str(key)] = float(valid.mean()) if len(valid) else 0.0
    return sum(value > random_top10 for value in values.values()), values


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    stage16 = args.stage16b_run.resolve()
    stage17 = args.stage17a_run.resolve()
    public_run = args.public_oof_run.resolve()
    validation_run = args.validation_run.resolve()
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    for run, label in [(stage17, "Stage 17A"), (validation_run, "Stage 21B")]:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        if summary.get("stage16b_manifest_sha256") != expected_hash:
            raise AssertionError(f"{label} manifest provenance mismatch")

    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cut_ids = pd.read_parquet(validation_run / "confidence_cut_report.parquet", columns=["cut_id"])
    cuts = pd.read_parquet(stage17 / "cut_report.parquet")
    selected = cuts[cuts["cut_id"].isin(cut_ids["cut_id"])].copy().sort_values("cut_id")
    if args.limit_cuts is not None:
        selected = selected.head(int(args.limit_cuts)).copy()
    assignments = pd.read_parquet(stage16 / "well_assignments.parquet")
    assignments["well_id"] = assignments["well_id"].astype(str)
    assignments["typewell_fold"] = _typewell_folds(assignments, 5, seed)
    selected = selected.merge(
        assignments[["well_id", "spatial_fold", "typewell_fold", "branch_group_fold"]],
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    selected["stage16_fold"] = selected["stage16_fold"].astype(int)
    wells = selected["well_id"].astype(str).unique().tolist()
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
    base_name = str(state_config.get("base_candidate", "top_pf_a130"))
    primary_variant = str(config.get("validation", {}).get("primary_variant", "ncc_mix"))
    profiles = [dict(value) for value in config.get("decoder_profiles", [])]
    primary_profile = str(config.get("validation", {}).get("primary_decoder", "medium"))
    row_parts: list[pd.DataFrame] = []
    invariance: list[bool] = []
    for index, cut in enumerate(selected.itertuples(index=False), 1):
        well_id, outer = str(cut.well_id), int(cut.cut_index)
        horizontal, typewell = load_well(well_id), load_typewell(well_id)
        source = public_by_well[well_id]
        original = int(source["row_index"].min())
        candidates = build_candidates(
            horizontal,
            typewell,
            outer,
            source["y_pred"].to_numpy(float)[outer - original :],
            candidate_config,
        )
        base = candidates[base_name]
        positions, surface, offsets, costs = strong_base_costs(
            horizontal, typewell, outer, base, state_config
        )
        truth = horizontal.iloc[positions]["TVT"].to_numpy(float)
        true_offset = truth - surface
        state = np.argmin(np.abs(offsets[None, :] - true_offset[:, None]), axis=1)
        in_grid = np.isfinite(true_offset) & (true_offset >= offsets[0]) & (true_offset <= offsets[-1])
        cost = costs[primary_variant]
        true_cost = cost[np.arange(len(cost)), state]
        valid = in_grid & (true_cost < 2.999)
        rank = 1 + np.sum(cost < true_cost[:, None], axis=1)
        part = pd.DataFrame(
            {
                "well_id": well_id,
                "cut_id": str(cut.cut_id),
                "requested_fraction": float(cut.requested_fraction),
                **{family: int(getattr(cut, family)) for family in FOLD_FAMILIES},
                "surface": surface,
                "truth": truth,
                "true_offset": true_offset,
                "oracle": surface + offsets[state],
                "offset_in_grid": in_grid,
                "emission_valid": valid,
                "rank": rank,
                "top5": valid & (rank <= 5),
                "top10": valid & (rank <= 10),
            }
        )
        for profile in profiles:
            decoded = decode_lattice(-cost, offsets, profile)
            correction = float(profile.get("blend_weight", 1.0)) * np.asarray(
                decoded["posterior_mean"], float
            )
            part[f"decoded_{profile['name']}"] = surface + correction
        row_parts.append(part)
        if len(invariance) < 8:
            changed = horizontal.copy()
            changed.loc[changed.index >= outer, "TVT"] += 9999.0
            changed_candidates = build_candidates(
                changed,
                typewell,
                outer,
                source["y_pred"].to_numpy(float)[outer - original :],
                candidate_config,
            )
            changed_state = strong_base_costs(
                changed, typewell, outer, changed_candidates[base_name], state_config
            )
            invariance.append(
                np.array_equal(surface, changed_state[1])
                and all(np.array_equal(costs[name], changed_state[3][name]) for name in costs)
            )
        if index % 10 == 0:
            print(f"strong-base NCC {index}/{len(selected)} cuts", flush=True)

    rows = pd.concat(row_parts, ignore_index=True)
    cut_rows = []
    for cut_id, group in rows.groupby("cut_id", sort=True):
        first = group.iloc[0]
        item: dict[str, Any] = {
            "cut_id": cut_id,
            "well_id": str(first["well_id"]),
            "requested_fraction": float(first["requested_fraction"]),
            **{family: int(first[family]) for family in FOLD_FAMILIES},
            "suffix_rows": len(group),
            "base_sse": float(np.square(group["surface"] - group["truth"]).sum()),
            "oracle_sse": float(np.square(group["oracle"] - group["truth"]).sum()),
        }
        for profile in profiles:
            name = str(profile["name"])
            item[f"candidate_sse_{name}"] = float(
                np.square(group[f"decoded_{name}"] - group["truth"]).sum()
            )
        cut_rows.append(item)
    report = pd.DataFrame.from_records(cut_rows)
    report["candidate_sse"] = report[f"candidate_sse_{primary_profile}"]
    metrics = _metrics(report)
    bootstrap = _bootstrap(
        report, int(config.get("validation", {}).get("bootstrap_resamples", 3000)), seed
    )
    total = len(rows)
    surface_rmse = float(np.sqrt(np.square(rows["surface"] - rows["truth"]).sum() / total))
    oracle_rmse = float(np.sqrt(np.square(rows["oracle"] - rows["truth"]).sum() / total))
    profile_report = []
    for profile in profiles:
        name = str(profile["name"])
        rmse = float(np.sqrt(np.square(rows[f"decoded_{name}"] - rows["truth"]).sum() / total))
        profile_report.append(
            {"profile": name, "rmse": rmse, "rmse_delta": rmse - surface_rmse}
        )
    valid = rows["emission_valid"].to_numpy(bool)
    valid_rank = rows.loc[valid, "rank"].to_numpy(float)
    top5 = float(rows.loc[valid, "top5"].mean()) if valid.any() else 0.0
    top10 = float(rows.loc[valid, "top10"].mean()) if valid.any() else 0.0
    random_top10 = 10.0 / len(offsets)
    signal_counts, signal_reports = {}, {}
    for family in FOLD_FAMILIES:
        count, values = _signal_counts(rows, family, random_top10)
        signal_counts[family], signal_reports[family] = count, values
    fraction_count, fraction_values = _signal_counts(rows, "requested_fraction", random_top10)
    validation_config = dict(config.get("validation", {}))
    gates = {
        "hidden_target_invariance": bool(invariance) and all(invariance),
        "offset_coverage": float(rows["offset_in_grid"].mean())
        >= float(validation_config.get("minimum_offset_coverage", 0.95)),
        "oracle_headroom": oracle_rmse - surface_rmse
        <= -float(validation_config.get("minimum_oracle_gain", 3.0)),
        "raw_top10_signal": top10
        >= random_top10 * float(validation_config.get("minimum_top10_random_multiplier", 1.25)),
        "raw_rank_signal": (float(np.median(valid_rank)) if len(valid_rank) else len(offsets))
        <= float(validation_config.get("maximum_median_rank", 25.0)),
        "standard_signal_consistency": signal_counts["stage16_fold"]
        >= int(validation_config.get("minimum_standard_signal_folds", 4)),
        "spatial_signal_consistency": signal_counts["spatial_fold"]
        >= int(validation_config.get("minimum_spatial_signal_folds", 4)),
        "typewell_signal_consistency": signal_counts["typewell_fold"]
        >= int(validation_config.get("minimum_typewell_signal_folds", 4)),
        "branch_signal_consistency": signal_counts["branch_group_fold"]
        >= int(validation_config.get("minimum_branch_signal_folds", 4)),
        "fraction_signal_consistency": fraction_count
        >= int(validation_config.get("minimum_signal_fractions", 3)),
    }
    promoted = bool(all(gates.values()))
    rows.to_parquet(output / "state_row_report.parquet", index=False)
    report.to_parquet(output / "state_cut_report.parquet", index=False)
    summary = {
        "stage23a_complete": True,
        "promoted_to_stage23b_learned_emission": promoted,
        "stage16b_manifest_sha256": expected_hash,
        "cuts": int(rows["cut_id"].nunique()),
        "wells": int(rows["well_id"].nunique()),
        "rows": len(rows),
        "offset_states": len(offsets),
        "offset_coverage": float(rows["offset_in_grid"].mean()),
        "emission_valid_fraction": float(valid.mean()),
        "surface_rmse": surface_rmse,
        "oracle_rmse": oracle_rmse,
        "oracle_delta": oracle_rmse - surface_rmse,
        "median_true_state_rank": float(np.median(valid_rank)) if len(valid_rank) else float(len(offsets)),
        "top5_recall": top5,
        "top10_recall": top10,
        "random_top10_recall": random_top10,
        "profile_report": profile_report,
        "primary_decoder_metrics": metrics,
        "primary_decoder_bootstrap_95pct": bootstrap,
        "signal_fold_counts": signal_counts,
        "signal_reports": signal_reports,
        "fraction_signal_count": fraction_count,
        "fraction_signal_report": fraction_values,
        "gates": gates,
        "next_step": (
            "Train a strong-base-aligned learned emission on a separate training split."
            if promoted
            else "Reject GR offset-state alignment and move to a non-GR physical state."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16),
        "stage17a_run": str(stage17),
        "public_oof_run": str(public_run),
        "validation_run": str(validation_run),
        "data_dir": str(args.data_dir.resolve()),
        "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
