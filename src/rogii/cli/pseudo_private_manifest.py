from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.testlike_validation import _frame_hash
from rogii.cli.trajectory_residual import _typewell_folds
from rogii.config import load_config


FOLD_COLUMNS = ["fold", "spatial_fold", "typewell_fold", "branch_group_fold"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze the Stage 37 pseudo-private benchmark manifest")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--split-run", type=Path, required=True)
    parser.add_argument("--design-validation-run", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _role_report(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "cuts": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "rows": int(frame["suffix_rows"].sum()),
        "fraction_counts": {
            f"{float(fraction):.2f}": int(count)
            for fraction, count in frame.groupby("requested_fraction", sort=True).size().items()
        },
        "fold_counts": {
            column: {str(int(fold)): int(count) for fold, count in frame.groupby(column, sort=True).size().items()}
            for column in FOLD_COLUMNS
        },
    }


def _load_cut_ids(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"cut_id", "well_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing columns: {missing}")
    frame["cut_id"] = frame["cut_id"].astype(str)
    frame["well_id"] = frame["well_id"].astype(str)
    return frame


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    stage16 = args.stage16b_run.resolve()
    stage17 = args.stage17a_run.resolve()
    split_run = args.split_run.resolve()
    design_run = args.design_validation_run.resolve()
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Stage 37 manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)

    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    stage17_summary = json.loads((stage17 / "summary.json").read_text(encoding="utf-8"))
    if stage17_summary.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 17A is not aligned to the frozen Stage 16B manifest")

    split_summary = json.loads((split_run / "summary.json").read_text(encoding="utf-8"))
    if not split_summary.get("public_replay_eligible_only"):
        raise AssertionError("Stage 37 requires a replay-eligible locked split")
    training_seed = _load_cut_ids(split_run / "training_cut_ids.parquet")
    confirmation_seed = _load_cut_ids(split_run / "confirmation_cut_ids.parquet")
    design_ids = _load_cut_ids(design_run / "confidence_cut_report.parquet")

    training_wells = set(training_seed["well_id"])
    confirmation_wells = set(confirmation_seed["well_id"])
    design_wells = set(design_ids["well_id"])
    overlaps = {
        "training_design": sorted(training_wells & design_wells),
        "training_confirmation": sorted(training_wells & confirmation_wells),
        "design_confirmation": sorted(design_wells & confirmation_wells),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Stage 37 split leakage: {overlaps}")

    cuts = pd.read_parquet(stage17 / "cut_report.parquet")
    required = {
        "cut_id", "well_id", "cut_index", "requested_fraction", "evaluation_role",
        "replay_eligible", "suffix_rows",
    }
    missing = sorted(required.difference(cuts.columns))
    if missing:
        raise KeyError(f"Stage 17 cut report is missing columns: {missing}")
    cuts["cut_id"] = cuts["cut_id"].astype(str)
    cuts["well_id"] = cuts["well_id"].astype(str)
    eligible = cuts[
        (cuts["evaluation_role"] == "primary") & cuts["replay_eligible"].astype(bool)
    ].copy()
    training = eligible[eligible["well_id"].isin(training_wells)].copy()
    design = eligible[eligible["cut_id"].isin(set(design_ids["cut_id"]))].copy()
    confirmation = eligible[eligible["cut_id"].isin(set(confirmation_seed["cut_id"]))].copy()
    if set(design["cut_id"]) != set(design_ids["cut_id"]):
        missing_design = sorted(set(design_ids["cut_id"]) - set(design["cut_id"]))
        raise AssertionError(f"Design cuts are not replay eligible: {missing_design[:5]}")
    if set(confirmation["cut_id"]) != set(confirmation_seed["cut_id"]):
        missing_confirmation = sorted(set(confirmation_seed["cut_id"]) - set(confirmation["cut_id"]))
        raise AssertionError(f"Confirmation cuts are not replay eligible: {missing_confirmation[:5]}")

    assignments = pd.read_parquet(stage16 / "well_assignments.parquet")
    assignments["well_id"] = assignments["well_id"].astype(str)
    assignments["typewell_fold"] = _typewell_folds(
        assignments,
        int(config.get("validation", {}).get("n_typewell_folds", 5)),
        int(config.get("seed", 42)),
    )
    assignment_columns = ["well_id", *FOLD_COLUMNS, "branch_group"]
    missing_assignments = sorted(set(assignment_columns).difference(assignments.columns))
    if missing_assignments:
        raise KeyError(f"Stage 16 assignments are missing columns: {missing_assignments}")

    parts = []
    for role, frame in (("training", training), ("design_validation", design), ("confirmation_locked", confirmation)):
        selected = frame[[
            "cut_id", "well_id", "cut_index", "requested_fraction", "suffix_rows",
            "evaluation_role", "replay_eligible",
        ]].copy()
        selected["benchmark_role"] = role
        parts.append(selected)
    manifest = pd.concat(parts, ignore_index=True).merge(
        assignments[assignment_columns],
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    if manifest[FOLD_COLUMNS].isna().any().any():
        raise AssertionError("Stage 37 fold assignments are incomplete")
    manifest = manifest.sort_values(
        ["benchmark_role", "well_id", "requested_fraction", "cut_id"], kind="stable"
    ).reset_index(drop=True)
    if manifest["cut_id"].duplicated().any():
        duplicated = manifest.loc[manifest["cut_id"].duplicated(), "cut_id"].tolist()
        raise AssertionError(f"Stage 37 roles share cuts: {duplicated[:5]}")

    expected = dict(config.get("expected", {}))
    role_frames = {
        role: manifest[manifest["benchmark_role"] == role].copy()
        for role in ("training", "design_validation", "confirmation_locked")
    }
    role_reports = {role: _role_report(frame) for role, frame in role_frames.items()}
    primary_fractions = {round(float(value), 2) for value in config.get("primary_fractions", [])}
    observed_training_fractions = {
        round(float(value), 2) for value in role_frames["training"]["requested_fraction"].unique()
    }
    confirmation_target_columns_read = False
    gates = {
        "stage16b_manifest_matches": True,
        "public_replay_eligible_only": bool(manifest["replay_eligible"].all()),
        "training_wells_match": role_reports["training"]["wells"] == int(expected["training_wells"]),
        "design_wells_match": role_reports["design_validation"]["wells"] == int(expected["design_wells"]),
        "design_cuts_match": role_reports["design_validation"]["cuts"] == int(expected["design_cuts"]),
        "confirmation_wells_match": role_reports["confirmation_locked"]["wells"] == int(expected["confirmation_wells"]),
        "split_overlap_zero": not any(overlaps.values()),
        "primary_fraction_coverage": primary_fractions.issubset(observed_training_fractions),
        "fold_assignments_complete": not manifest[FOLD_COLUMNS].isna().any().any(),
        "confirmation_target_columns_unread": not confirmation_target_columns_read,
    }
    manifest.to_parquet(output / "pseudo_private_manifest.parquet", index=False)
    assignments[
        assignments["well_id"].isin(set(manifest["well_id"]))
    ][assignment_columns].sort_values("well_id").to_parquet(
        output / "benchmark_well_assignments.parquet", index=False
    )
    manifest_hash = _frame_hash(manifest)
    summary = {
        "stage37a_complete": True,
        "promoted_to_stage37b_top_pf_replay": bool(all(gates.values())),
        "stage16b_manifest_sha256": expected_hash,
        "pseudo_private_manifest_sha256": manifest_hash,
        "manifest_file_sha256": _sha256(output / "pseudo_private_manifest.parquet"),
        "roles": role_reports,
        "overlaps": overlaps,
        "confirmation_locked": True,
        "confirmation_target_columns_read": confirmation_target_columns_read,
        "gates": gates,
        "next_step": (
            "Materialize target-safe top-PF proxy predictions for training and design roles only."
            if all(gates.values())
            else "Repair the split contract before computing any top-PF predictions."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16),
        "stage17a_run": str(stage17),
        "split_run": str(split_run),
        "design_validation_run": str(design_run),
        "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
