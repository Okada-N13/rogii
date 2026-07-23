from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config


SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {
    0: "pf_scale_5_hold_0.2",
    1: "pf_scale_3_hold_0.15",
    2: "pf_scale_12_beam_0.2_hold_0.15",
    3: "pf_scale_5_hold_0.15",
    4: "pf_scale_5_beam_0.05_hold_0.05",
    5: "pf_scale_12_beam_0.2_hold_0.05",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screen the SP45 likelihood PF on uncovered Stage 17 cuts")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit-wells", type=int)
    return parser


def selector_variant(n_eval: int, z_values: np.ndarray) -> tuple[int, str]:
    finite = np.asarray(z_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    z_span = float(np.ptp(finite)) if len(finite) else 0.0
    code = int(float(n_eval) > SELECTOR_N_EVAL_THRESHOLD) + 2 * int(
        np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right")
    )
    return code, SELECTOR_BIN_VARIANTS.get(code, "pf_scale_8_hold_0.2")


def parse_selector_variant(name: str) -> tuple[float, float, float]:
    parts = name.split("_")
    scale = float(parts[2])
    beam = float(parts[parts.index("beam") + 1]) if "beam" in parts else 0.0
    hold = float(parts[parts.index("hold") + 1]) if "hold" in parts else 0.0
    return scale, beam, hold


@njit(cache=True)
def _interp_grid(grid: np.ndarray, value: float, minimum: float, step: float) -> float:
    position = (value - minimum) / step
    index = int(position)
    if index < 0:
        return grid[0]
    if index >= len(grid) - 1:
        return grid[-1]
    fraction = position - index
    return grid[index] * (1.0 - fraction) + grid[index + 1] * fraction


@njit(cache=True, nogil=True)
def _particle_paths(
    md: np.ndarray, z: np.ndarray, gr: np.ndarray, grid: np.ndarray,
    grid_minimum: float, grid_step: float, gr_sigma: float, level_start: float,
    initial_rate: float, particles: int, seeds: int, seed_base: int,
) -> tuple[np.ndarray, np.ndarray]:
    paths = np.empty((seeds, len(md)), dtype=np.float64)
    likelihoods = np.empty(seeds, dtype=np.float64)
    maximum = grid_minimum + len(grid) * grid_step
    for seed in range(seeds):
        np.random.seed(seed_base + seed)
        position = np.empty(particles, dtype=np.float64)
        rate = np.empty(particles, dtype=np.float64)
        weight = np.ones(particles, dtype=np.float64) / particles
        for item in range(particles):
            position[item] = level_start + 4.5 * np.random.randn()
            rate[item] = initial_rate + 0.01 * np.random.randn()
        log_likelihood = 0.0
        previous_md = md[0] - 1.0
        for row in range(len(md)):
            md_step = max(md[row] - previous_md, 1.0)
            for item in range(particles):
                rate[item] = 0.998 * rate[item] + 0.002 * np.random.randn()
                position[item] += rate[item] * md_step + 0.005 * np.random.randn()
                tvt = min(max(position[item] - z[row], grid_minimum - 100.0), maximum + 100.0)
                position[item] = tvt + z[row]
            average_likelihood = 0.0
            for item in range(particles):
                expected_gr = _interp_grid(grid, position[item] - z[row], grid_minimum, grid_step)
                delta = (gr[row] - expected_gr) / gr_sigma
                likelihood = np.exp(-0.5 * min(delta * delta, 600.0))
                likelihood = max(likelihood, 1e-300)
                average_likelihood += weight[item] * likelihood
                weight[item] *= likelihood
            log_likelihood += np.log(max(average_likelihood, 1e-300))
            weight_sum = weight.sum()
            if weight_sum > 0.0:
                weight /= weight_sum
            else:
                weight[:] = 1.0 / particles
            effective = 1.0 / np.square(weight).sum()
            if effective < 0.5 * particles:
                cumulative = np.cumsum(weight)
                start = np.random.uniform(0.0, 1.0 / particles)
                new_position = np.empty(particles, dtype=np.float64)
                new_rate = np.empty(particles, dtype=np.float64)
                cursor = 0
                for item in range(particles):
                    target = start + item / particles
                    while cursor < particles - 1 and cumulative[cursor] < target:
                        cursor += 1
                    new_position[item] = position[cursor] + 0.1 * np.random.randn()
                    new_rate[item] = rate[cursor] + 0.001 * np.random.randn()
                position, rate = new_position, new_rate
                weight[:] = 1.0 / particles
            estimate = 0.0
            for item in range(particles):
                estimate += weight[item] * (position[item] - z[row])
            paths[seed, row] = estimate
            previous_md = md[row]
        likelihoods[seed] = log_likelihood
    return paths, likelihoods


def _sample_positions(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, maximum).round().astype(np.int64))


def likelihood_selector(
    horizontal: pd.DataFrame, typewell: pd.DataFrame, cut_index: int, config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    prefix = horizontal.iloc[:cut_index]
    suffix = horizontal.iloc[cut_index:]
    if len(prefix) < 3 or len(suffix) == 0:
        raise ValueError("Selector requires a non-empty suffix and at least three prefix rows")
    tw = typewell[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna().sort_values("TVT")
    if len(tw) < 3:
        raise ValueError("Typewell has fewer than three valid rows")
    tw_tvt, tw_gr = tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float)
    grid_step = float(config.get("typewell_grid_step", 0.2))
    grid_tvt = np.arange(float(tw_tvt.min()), float(tw_tvt.max()) + grid_step, grid_step)
    grid = np.interp(grid_tvt, tw_tvt, tw_gr).astype(np.float64)
    prefix_tvt = prefix["TVT"].to_numpy(float)
    prefix_gr = pd.to_numeric(prefix["GR"], errors="coerce").interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    reference_gr = np.interp(prefix_tvt, tw_tvt, tw_gr)
    gr_sigma = float(np.clip(np.nanstd(prefix_gr - reference_gr), 10.0, 60.0))
    # The top-PF A130 branch scales the visible-prefix GR likelihood width by 1.30.
    # Keep 1.0 as the historical Stage 17 default and expose the multiplier for
    # target-safe replay of the actual 6.589 submission family.
    gr_sigma *= float(config.get("gr_sigma_multiplier", 1.0))
    tail = prefix.tail(30)
    md_delta = np.diff(tail["MD"].to_numpy(float))
    level_delta = np.diff(tail["TVT"].to_numpy(float) + tail["Z"].to_numpy(float))
    valid = md_delta > 0
    initial_rate = float(np.median(level_delta[valid] / md_delta[valid])) if valid.sum() >= 3 else 0.0
    full_md = suffix["MD"].to_numpy(float)
    full_z = suffix["Z"].to_numpy(float)
    full_gr = pd.to_numeric(horizontal["GR"], errors="coerce").interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)[cut_index:]
    sampled = _sample_positions(len(suffix), int(config.get("maximum_tracking_steps", 512)))
    paths, likelihoods = _particle_paths(
        full_md[sampled], full_z[sampled], full_gr[sampled], grid, float(grid_tvt[0]), grid_step,
        gr_sigma, float(prefix_tvt[-1] + prefix["Z"].iloc[-1]), initial_rate,
        int(config.get("particles", 96)), int(config.get("seeds", 8)), int(config.get("seed_base", 0)),
    )
    code, variant = selector_variant(len(suffix), full_z)
    scale, beam_weight, hold_weight = parse_selector_variant(variant)
    centered = likelihoods - float(likelihoods.max())
    weights = np.exp(centered / scale)
    weights /= weights.sum()
    coarse = (weights[:, None] * paths).sum(axis=0)
    prediction = np.interp(full_md, full_md[sampled], coarse)
    last_tvt = float(prefix_tvt[-1])
    # Stage 17B screens the dominant likelihood-PF signal. Beam weights are recorded but
    # intentionally not applied until this branch proves useful on uncovered cuts.
    prediction = (1.0 - hold_weight) * prediction + hold_weight * last_tvt
    return prediction, {
        "selector_code": code, "selector_variant": variant, "scale": scale,
        "configured_beam_weight": beam_weight, "hold_weight": hold_weight,
        "tracking_steps": len(sampled), "suffix_rows": len(suffix),
        "particles": int(config.get("particles", 96)), "seeds": int(config.get("seeds", 8)),
        "gr_sigma": gr_sigma, "likelihood_spread": float(likelihoods.std()),
    }


def _rmse(sse: float, rows: int) -> float:
    return float(np.sqrt(float(sse) / int(rows))) if rows else float("nan")


def _role_report(cuts: pd.DataFrame, role: str) -> dict[str, Any]:
    selected = cuts[cuts["evaluation_role"] == role]
    uncovered = selected[~selected["replay_eligible"]]
    rows = int(selected["suffix_rows"].sum())
    uncovered_rows = int(uncovered["suffix_rows"].sum())
    baseline = _rmse(float(selected["baseline_sse"].sum()), rows)
    stage17a = _rmse(float(selected["hybrid_sse"].sum()), rows)
    candidate = _rmse(float(selected["candidate_sse"].sum()), rows)
    uncovered_base = _rmse(float(uncovered["baseline_sse"].sum()), uncovered_rows)
    selector = _rmse(float(uncovered["selector_sse"].sum()), uncovered_rows)
    return {
        "rows": rows, "uncovered_rows": uncovered_rows,
        "uncovered_row_fraction": float(uncovered_rows / rows) if rows else 0.0,
        "baseline_rmse": baseline, "stage17a_hybrid_rmse": stage17a,
        "candidate_rmse": candidate, "delta_vs_stage17a": candidate - stage17a,
        "uncovered_baseline_rmse": uncovered_base, "uncovered_selector_rmse": selector,
        "uncovered_selector_delta": selector - uncovered_base,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    stage16, stage17a = args.stage16b_run.resolve(), args.stage17a_run.resolve()
    manifest = pd.read_parquet(stage16 / "pseudo_test_manifest.parquet")
    cuts = pd.read_parquet(stage17a / "cut_report.parquet")
    stage17_summary = json.loads((stage17a / "summary.json").read_text(encoding="utf-8"))
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    if stage17_summary.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 17A did not use the frozen Stage 16B manifest")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    selected_wells = sorted(cuts["well_id"].astype(str).unique())
    if args.limit_wells is not None:
        selected_wells = selected_wells[: int(args.limit_wells)]
        cuts = cuts[cuts["well_id"].astype(str).isin(selected_wells)].copy()
        manifest = manifest[manifest["well_id"].astype(str).isin(selected_wells)]
    selector_config = dict(config.get("selector", {}))
    manifest_index = manifest.set_index("cut_id")
    writer: pq.ParquetWriter | None = None
    prediction_path = output / "selector_predictions.parquet"
    audit_rows: list[dict[str, Any]] = []
    cuts["selector_sse"] = np.nan
    cuts["candidate_sse"] = cuts["hybrid_sse"].astype(float)
    uncovered_indices = cuts.index[~cuts["replay_eligible"]].tolist()
    try:
        for position, index in enumerate(uncovered_indices, 1):
            cut = cuts.loc[index]
            well_id, cut_id = str(cut["well_id"]), str(cut["cut_id"])
            details = manifest_index.loc[cut_id]
            horizontal = pd.read_csv(args.data_dir.resolve() / "train" / f"{well_id}__horizontal_well.csv")
            typewell = pd.read_csv(args.data_dir.resolve() / "train" / f"{well_id}__typewell.csv")
            cut_index = int(cut["cut_index"])
            model_input = horizontal[["MD", "Z", "GR", "TVT"]].copy()
            model_input.loc[model_input.index >= cut_index, "TVT"] = np.nan
            prediction, audit = likelihood_selector(model_input, typewell, cut_index, selector_config)
            truth = horizontal["TVT"].to_numpy(float)[cut_index:]
            selector_sse = float(np.square(prediction - truth).sum())
            cuts.loc[index, "selector_sse"] = selector_sse
            cuts.loc[index, "candidate_sse"] = selector_sse
            frame = pd.DataFrame({
                "cut_id": cut_id, "source_well_id": well_id,
                "row_index": np.arange(cut_index, len(horizontal), dtype=np.int32),
                "MD": horizontal["MD"].to_numpy(float)[cut_index:],
                "stage16_fold": np.int16(details["fold"]),
                "evaluation_role": str(cut["evaluation_role"]),
                "requested_fraction": np.float32(cut["requested_fraction"]),
                "y_true": truth, "y_pred": prediction,
                "baseline_y_pred": float(horizontal["TVT"].iloc[cut_index - 1]),
                "residual_target": truth - prediction,
            })
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(prediction_path, table.schema, compression="zstd")
            writer.write_table(table)
            audit_rows.append({"cut_id": cut_id, "well_id": well_id, **audit})
            if position % 100 == 0:
                print(f"selector replay {position}/{len(uncovered_indices)} cuts", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if cuts.loc[~cuts["replay_eligible"], "selector_sse"].isna().any():
        raise AssertionError("Selector predictions are incomplete")
    cuts.to_parquet(output / "cut_report.parquet", index=False)
    pd.DataFrame.from_records(audit_rows).to_parquet(output / "selector_audit.parquet", index=False)
    reports = {role: _role_report(cuts, role) for role in ["primary", "diagnostic"]}
    fold_report = []
    for fold, frame in cuts[cuts["evaluation_role"] == "primary"].groupby("stage16_fold", sort=True):
        rows = int(frame["suffix_rows"].sum())
        old = _rmse(float(frame["hybrid_sse"].sum()), rows)
        new = _rmse(float(frame["candidate_sse"].sum()), rows)
        fold_report.append({"fold": int(fold), "stage17a_rmse": old, "candidate_rmse": new, "delta": new - old})
    improved_folds = sum(row["delta"] < 0 for row in fold_report)
    gates_config = dict(config.get("gates", {}))
    gates = {
        "hidden_target_invariance": True,
        "uncovered_selector_gain": reports["primary"]["uncovered_selector_delta"] <= -float(gates_config.get("minimum_uncovered_gain", 0.05)),
        "full_primary_gain": reports["primary"]["delta_vs_stage17a"] <= -float(gates_config.get("minimum_full_gain", 0.02)),
        "fold_consistency": improved_folds >= int(gates_config.get("minimum_improved_folds", 4)),
    }
    summary = {
        "stage17b_complete": True, "promoted_to_full_selector_validation": bool(all(gates.values())),
        "stage16b_manifest_sha256": expected_hash,
        "selector_profile": selector_config, "role_report": reports,
        "fold_report": fold_report, "improved_folds": f"{improved_folds}/{len(fold_report)}",
        "n_uncovered_cuts": len(uncovered_indices), "gates": gates,
        "approximation_note": "Likelihood PF only; suffix capped and interpolated; configured beam weight not applied.",
        "next_step": "Validate full-resolution selector and beam only if this screen promotes.",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {"stage16b_run": str(stage16), "stage17a_run": str(stage17a), "run_id": args.run_id, "limit_wells": args.limit_wells}
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
