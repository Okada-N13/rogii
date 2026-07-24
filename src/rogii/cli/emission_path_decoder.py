from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.branch_retrieval import _bootstrap, _metrics
from rogii.cli.emission_decoder import _load_checkpoint
from rogii.cli.emission_hierarchical_decoder import _apply_correction, _cut_report, _p90_delta
from rogii.cli.prefix_router import FOLD_FAMILIES, _family_report
from rogii.cli.strong_base_emission import _device, _softmax, build_strong_base_sequence
from rogii.cli.trajectory_residual import _typewell_folds
from rogii.config import load_config
from rogii.models.emission_tcn import predict_emissions
from rogii.models.raw_ncc import offset_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 25A temporal emission path decoder")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--public-oof-run", type=Path, required=True)
    parser.add_argument("--stage24a-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def _centered_smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1 or len(values) <= 1:
        return np.asarray(values, float).copy()
    radius = window // 2
    padded = np.pad(np.asarray(values, float), (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def viterbi_offsets(
    logits: np.ndarray,
    offsets: np.ndarray,
    transition_weight: float,
    anchor_weight: float,
) -> np.ndarray:
    logits = np.asarray(logits, float)
    offsets = np.asarray(offsets, float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    emission = -shifted + float(anchor_weight) * np.abs(offsets)[None, :]
    transition = float(transition_weight) * np.square(
        offsets[:, None] - offsets[None, :]
    )
    steps, states = logits.shape
    back = np.zeros((steps, states), dtype=np.int16)
    score = emission[0]
    for step in range(1, steps):
        candidates = score[:, None] + transition
        previous = np.argmin(candidates, axis=0)
        back[step] = previous
        score = candidates[previous, np.arange(states)] + emission[step]
    path = np.empty(steps, dtype=np.int64)
    path[-1] = int(np.argmin(score))
    for step in range(steps - 1, 0, -1):
        path[step - 1] = back[step, path[step]]
    return offsets[path]


def decode_profile(profile: dict[str, Any], logits: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    probability = _softmax(logits)
    kind = str(profile["kind"])
    if kind == "smooth_mean":
        raw = probability @ offsets
        decoded = _centered_smooth(raw, int(profile["window"]))
    elif kind == "posterior_median":
        cumulative = np.cumsum(probability, axis=1)
        decoded = offsets[np.argmax(cumulative >= 0.5, axis=1)]
    elif kind == "viterbi":
        decoded = viterbi_offsets(
            logits,
            offsets,
            float(profile["transition_weight"]),
            float(profile["anchor_weight"]),
        )
    else:
        raise ValueError(f"Unknown path decoder kind: {kind}")
    return float(profile["decoder_weight"]) * np.asarray(decoded, float)


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
    if summary24.get("promoted_to_stage24b_reserved_confirmation") is not False:
        raise AssertionError("Stage 25A is the decoder screen for rejected Stage 24A")
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
    sequences = []
    branch_by_cut = {}
    for index, record in enumerate(validation.itertuples(index=False), 1):
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
        sequences.append(
            build_strong_base_sequence(
                record,
                horizontal,
                load_typewell(well_id),
                public_prediction,
                candidate_config,
                state_config,
            )
        )
        branch_by_cut[str(record.cut_id)] = int(record.branch_group_fold)
        if index % 10 == 0:
            print(f"path decoder volumes {index}/{len(validation)} cuts", flush=True)

    offsets = offset_grid(state_config)
    device = _device(args.device)
    folds = list(range(int(summary24["ensemble_models"])))
    ensemble = [
        np.zeros((len(item.target_state), len(offsets)), dtype=np.float32)
        for item in sequences
    ]
    for fold in folds:
        model = _load_checkpoint(stage24 / f"fold_{fold}.pt", offsets, device)
        predictions = predict_emissions(model, sequences, device)
        for target, values in zip(ensemble, predictions, strict=True):
            target += values.astype(np.float32) / len(folds)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    base_parts = []
    for item in sequences:
        base_parts.append(
            pd.DataFrame(
                {
                    "well_id": item.well_id,
                    "cut_id": item.cut_id,
                    "requested_fraction": item.cut_fraction,
                    "stage16_fold": item.fold,
                    "spatial_fold": item.spatial_fold,
                    "typewell_fold": item.typewell_fold,
                    "branch_group_fold": int(branch_by_cut[item.cut_id]),
                    "surface": item.surface_y_pred,
                    "truth": item.y_true,
                }
            )
        )
    rows = pd.concat(base_parts, ignore_index=True)
    profiles = [dict(value) for value in config.get("decoder_profiles", [])]
    correction_config = dict(config.get("correction", {}))
    screen_config = dict(config.get("screen", {}))
    profile_reports = []
    for profile in profiles:
        name = str(profile["name"])
        prediction_parts = []
        for item, logits in zip(sequences, ensemble, strict=True):
            correction = decode_profile(profile, logits, offsets)
            prediction_parts.append(
                _apply_correction(
                    item.surface_y_pred,
                    correction,
                    float(correction_config.get("cap_ft", 12.0)),
                    float(correction_config.get("ramp_rows", 64.0)),
                )
            )
        rows[f"prediction_{name}"] = np.concatenate(prediction_parts)
        report = _cut_report(rows, f"prediction_{name}")
        metrics = _metrics(report)
        bootstrap = _bootstrap(
            report, int(screen_config.get("bootstrap_resamples", 3000)), seed
        )
        p90 = _p90_delta(report)
        family_reports = {
            family: _family_report(report, family) for family in FOLD_FAMILIES
        }
        eligible = (
            metrics["rmse_delta"] <= -float(screen_config.get("minimum_gain", 0.1))
            and bootstrap[1] < 0
            and (
                not bool(screen_config.get("require_p90_nonworse", True))
                or p90 <= 0
            )
            and metrics["improved_folds"]
            >= int(screen_config.get("minimum_improved_folds", 4))
            and metrics["improved_fractions"]
            >= int(screen_config.get("minimum_improved_fractions", 3))
            and family_reports["spatial_fold"]["improved_folds"]
            >= int(screen_config.get("minimum_spatial_folds", 4))
            and family_reports["typewell_fold"]["improved_folds"]
            >= int(screen_config.get("minimum_typewell_folds", 4))
            and family_reports["branch_group_fold"]["improved_folds"]
            >= int(screen_config.get("minimum_branch_folds", 4))
        )
        profile_reports.append(
            {
                "profile": name,
                "kind": str(profile["kind"]),
                "rmse": metrics["candidate_rmse"],
                "rmse_delta": metrics["rmse_delta"],
                "p90_delta": p90,
                "bootstrap_95pct": bootstrap,
                "improved_folds": metrics["improved_folds"],
                "improved_fractions": metrics["improved_fractions"],
                "spatial_improved_folds": family_reports["spatial_fold"]["improved_folds"],
                "typewell_improved_folds": family_reports["typewell_fold"]["improved_folds"],
                "branch_improved_folds": family_reports["branch_group_fold"]["improved_folds"],
                "eligible": eligible,
            }
        )
        report.to_parquet(output / f"cut_report_{name}.parquet", index=False)
    eligible = [value for value in profile_reports if value["eligible"]]
    selected = (
        min(eligible, key=lambda value: (value["rmse"], value["profile"]))["profile"]
        if eligible else None
    )
    rows.to_parquet(output / "design_validation_path_rows.parquet", index=False)
    summary = {
        "stage25a_complete": True,
        "promoted_to_stage25b_oof_audit": selected is not None,
        "stage16b_manifest_sha256": expected_hash,
        "device": str(device),
        "design_validation_cuts": len(sequences),
        "design_validation_wells": len(wells),
        "profiles": profile_reports,
        "selected_screen_profile": selected,
        "screen_role": "exploratory_design_validation_only",
        "reserved_confirmation_used": False,
        "next_step": (
            "Regenerate 500-cut OOF logits and run nested path-profile audit."
            if selected
            else "Reject temporal path decoding and redesign the state representation."
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
