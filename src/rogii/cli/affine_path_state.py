from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.prefix_router import FOLD_FAMILIES
from rogii.cli.strong_base_emission import build_strong_base_sequence
from rogii.cli.trajectory_residual import _typewell_folds
from rogii.config import load_config
from rogii.models.affine_path_state import (
    affine_path_library,
    aggregate_path_costs,
    decode_path_scores,
)
from rogii.models.raw_ncc import offset_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 26A affine path-state signal audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--public-oof-run", type=Path, required=True)
    parser.add_argument("--stage24a-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _group_signal(frame: pd.DataFrame, column: str, random_top10: float) -> dict[str, object]:
    report = {}
    for key, group in frame.groupby(column, sort=True):
        recall = float(group["top10"].mean())
        report[str(key)] = {
            "cuts": len(group),
            "top10_recall": recall,
            "random_top10_recall": random_top10,
            "signal": recall > random_top10,
        }
    return {
        "signal_groups": sum(value["signal"] for value in report.values()),
        "groups": len(report),
        "report": report,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    stage16 = args.stage16b_run.resolve()
    stage17 = args.stage17a_run.resolve()
    public_run = args.public_oof_run.resolve()
    stage24 = args.stage24a_run.resolve()
    validation_run = args.validation_run.resolve()
    summary24 = json.loads((stage24 / "summary.json").read_text(encoding="utf-8"))
    if summary24.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 24A manifest provenance mismatch")
    if not summary24.get("stage24a_complete"):
        raise AssertionError("Stage 24A did not complete")
    if not summary24.get("gates", {}).get("hidden_target_invariance"):
        raise AssertionError("Stage 24A hidden-target invariance failed")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_parquet(stage17 / "cut_report.parquet")
    validation_ids = pd.read_parquet(
        validation_run / "confidence_cut_report.parquet", columns=["cut_id"]
    )
    validation = cuts[cuts["cut_id"].isin(validation_ids["cut_id"])].copy().sort_values("cut_id")
    assignments = pd.read_parquet(stage16 / "well_assignments.parquet")
    assignments["well_id"] = assignments["well_id"].astype(str)
    assignments["typewell_fold"] = _typewell_folds(assignments, 5, seed)
    validation = validation.merge(
        assignments[["well_id", "spatial_fold", "typewell_fold", "branch_group_fold"]],
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    validation["stage16_fold"] = validation["stage16_fold"].astype(int)
    wells = sorted(validation["well_id"].astype(str).unique())
    public = pd.read_parquet(
        public_run / "base_oof.parquet",
        columns=["well_id", "row_index", "y_pred"],
        filters=[("well_id", "in", wells)],
    )
    public["well_id"] = public["well_id"].astype(str)
    public_by_well = {
        well: frame.sort_values("row_index")
        for well, frame in public.groupby("well_id", sort=True)
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
    path_config = dict(config.get("affine_paths", {}))
    endpoints = np.arange(
        float(path_config["endpoint_min_ft"]),
        float(path_config["endpoint_max_ft"]) + 0.5 * float(path_config["endpoint_step_ft"]),
        float(path_config["endpoint_step_ft"]),
    )
    offsets = offset_grid(state_config)
    profiles = [dict(value) for value in config.get("decoder_profiles", [])]
    rank_rows = []
    decoder_rows = []
    for position, record in enumerate(validation.itertuples(index=False), 1):
        well_id, outer = str(record.well_id), int(record.cut_index)
        horizontal = load_well(well_id)
        source = public_by_well[well_id]
        original = int(source["row_index"].min())
        source_index = source["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(source_index, np.arange(original, len(horizontal))):
            raise AssertionError(f"{well_id}: public OOF is not a contiguous suffix")
        if outer < original:
            raise AssertionError(f"{record.cut_id}: cut precedes public OOF start")
        public_prediction = source["y_pred"].to_numpy(float)[outer - original :]
        sequence = build_strong_base_sequence(
            record,
            horizontal,
            load_typewell(well_id),
            public_prediction,
            candidate_config,
            state_config,
        )
        corrections, specifications = affine_path_library(len(sequence.y_true), endpoints)
        scores, valid_fraction = aggregate_path_costs(
            sequence.costs[:, 3].astype(np.float32),
            offsets,
            corrections,
            float(path_config.get("invalid_cost", 2.999)),
            float(path_config.get("missing_penalty", 0.5)),
        )
        squared_error = np.square(
            sequence.surface_y_pred[:, None] + corrections - sequence.y_true[:, None]
        ).sum(axis=0)
        oracle = int(np.argmin(squared_error))
        rank = 1 + int(np.sum(scores < scores[oracle]))
        base_sse = float(np.square(sequence.surface_y_pred - sequence.y_true).sum())
        common = {
            "cut_id": sequence.cut_id,
            "well_id": sequence.well_id,
            "requested_fraction": sequence.cut_fraction,
            "stage16_fold": sequence.fold,
            "spatial_fold": sequence.spatial_fold,
            "typewell_fold": sequence.typewell_fold,
            "branch_group_fold": int(record.branch_group_fold),
            "rows": len(sequence.y_true),
        }
        rank_rows.append(
            {
                **common,
                "base_sse": base_sse,
                "oracle_sse": float(squared_error[oracle]),
                "oracle_rank": rank,
                "top5": rank <= 5,
                "top10": rank <= 10,
                "oracle_start_offset": float(specifications[oracle, 0]),
                "oracle_end_offset": float(specifications[oracle, 1]),
                "mean_valid_path_fraction": float(valid_fraction.mean()),
            }
        )
        item = dict(common)
        item["base_sse"] = base_sse
        for profile in profiles:
            correction = float(profile["decoder_weight"]) * decode_path_scores(
                scores,
                corrections,
                str(profile["kind"]),
                float(profile.get("temperature", 0.1)),
            )
            prediction = sequence.surface_y_pred + correction
            item[f"candidate_sse_{profile['name']}"] = float(
                np.square(prediction - sequence.y_true).sum()
            )
        decoder_rows.append(item)
        if position % 10 == 0:
            print(f"affine path states {position}/{len(validation)} cuts", flush=True)

    ranks = pd.DataFrame(rank_rows)
    decoders = pd.DataFrame(decoder_rows)
    total_rows = int(ranks["rows"].sum())
    base_rmse = float(np.sqrt(ranks["base_sse"].sum() / total_rows))
    oracle_rmse = float(np.sqrt(ranks["oracle_sse"].sum() / total_rows))
    states = len(endpoints) ** 2
    random_top10 = min(10, states) / states
    signal_reports = {
        family: _group_signal(ranks, family, random_top10)
        for family in (*FOLD_FAMILIES, "requested_fraction")
    }
    decoder_report = []
    for profile in profiles:
        column = f"candidate_sse_{profile['name']}"
        rmse = float(np.sqrt(decoders[column].sum() / total_rows))
        decoder_report.append(
            {
                "profile": str(profile["name"]),
                "kind": str(profile["kind"]),
                "rmse": rmse,
                "rmse_delta": rmse - base_rmse,
            }
        )
    gate_config = dict(config.get("gates", {}))
    gates = {
        "hidden_target_invariance": bool(summary24["gates"]["hidden_target_invariance"]),
        "oracle_headroom": oracle_rmse - base_rmse
        <= -float(gate_config.get("minimum_oracle_gain", 2.0)),
        "valid_path_coverage": float(ranks["mean_valid_path_fraction"].mean())
        >= float(gate_config.get("minimum_valid_path_fraction", 0.8)),
        "top10_rank_signal": float(ranks["top10"].mean())
        >= random_top10 * float(gate_config.get("minimum_top10_random_multiple", 1.5)),
        "median_rank_signal": float(ranks["oracle_rank"].median())
        <= float(gate_config.get("maximum_median_rank", 50)),
        "standard_signal": signal_reports["stage16_fold"]["signal_groups"]
        >= int(gate_config.get("minimum_standard_signal_folds", 4)),
        "spatial_signal": signal_reports["spatial_fold"]["signal_groups"]
        >= int(gate_config.get("minimum_spatial_signal_folds", 4)),
        "typewell_signal": signal_reports["typewell_fold"]["signal_groups"]
        >= int(gate_config.get("minimum_typewell_signal_folds", 4)),
        "branch_signal": signal_reports["branch_group_fold"]["signal_groups"]
        >= int(gate_config.get("minimum_branch_signal_folds", 4)),
        "fraction_signal": signal_reports["requested_fraction"]["signal_groups"]
        >= int(gate_config.get("minimum_fraction_signal_groups", 3)),
    }
    promoted = bool(all(gates.values()))
    ranks.to_parquet(output / "affine_path_rank_report.parquet", index=False)
    decoders.to_parquet(output / "affine_path_decoder_report.parquet", index=False)
    summary = {
        "stage26a_complete": True,
        "promoted_to_stage26b_learned_path_ranker": promoted,
        "stage16b_manifest_sha256": expected_hash,
        "cuts": len(ranks),
        "wells": int(ranks["well_id"].nunique()),
        "rows": total_rows,
        "endpoint_states": len(endpoints),
        "affine_path_states": states,
        "base_rmse": base_rmse,
        "oracle_rmse": oracle_rmse,
        "oracle_delta": oracle_rmse - base_rmse,
        "mean_valid_path_fraction": float(ranks["mean_valid_path_fraction"].mean()),
        "median_oracle_rank": float(ranks["oracle_rank"].median()),
        "top5_recall": float(ranks["top5"].mean()),
        "top10_recall": float(ranks["top10"].mean()),
        "random_top10_recall": random_top10,
        "signal_reports": signal_reports,
        "decoder_report": decoder_report,
        "gates": gates,
        "reserved_confirmation_used": False,
        "next_step": (
            "Train a cut-level affine path ranker on the 500 training cuts."
            if promoted
            else "Reject GR affine path states and move to a non-GR state representation."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16),
        "stage17a_run": str(stage17),
        "public_oof_run": str(public_run),
        "stage24a_run": str(stage24),
        "validation_run": str(validation_run),
        "data_dir": str(args.data_dir.resolve()),
        "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
