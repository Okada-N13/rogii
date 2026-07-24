from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from rogii.artifacts import environment_report, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 24A scaled emission split manifest")
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--design-validation-run", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-wells-per-fold", type=int, default=100)
    parser.add_argument("--confirmation-wells-per-fold", type=int, default=24)
    parser.add_argument("--target-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _stable_order(well_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{well_id}".encode()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stage17 = args.stage17a_run.resolve()
    validation_run = args.design_validation_run.resolve()
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_parquet(stage17 / "cut_report.parquet")
    validation_ids = pd.read_parquet(
        validation_run / "confidence_cut_report.parquet", columns=["cut_id"]
    )
    validation = cuts[cuts["cut_id"].isin(validation_ids["cut_id"])].copy()
    validation_wells = set(validation["well_id"].astype(str))
    primary = cuts[cuts["evaluation_role"] == "primary"].copy()
    primary["well_id"] = primary["well_id"].astype(str)
    primary = primary[~primary["well_id"].isin(validation_wells)].copy()
    primary["_fraction_distance"] = (
        primary["requested_fraction"].astype(float) - float(args.target_fraction)
    ).abs()
    one_per_well = (
        primary.sort_values(["well_id", "_fraction_distance", "cut_id"])
        .groupby("well_id", sort=True)
        .head(1)
        .copy()
    )
    one_per_well["_stable_order"] = one_per_well["well_id"].map(
        lambda value: _stable_order(value, int(args.seed))
    )

    training_parts = []
    confirmation_parts = []
    fold_report = []
    for fold, group in one_per_well.groupby("stage16_fold", sort=True):
        ordered = group.sort_values(["_stable_order", "well_id"])
        training = ordered.head(int(args.training_wells_per_fold))
        confirmation = ordered.iloc[
            int(args.training_wells_per_fold) :
            int(args.training_wells_per_fold) + int(args.confirmation_wells_per_fold)
        ]
        if len(training) != int(args.training_wells_per_fold):
            raise RuntimeError(f"Fold {fold} has only {len(training)} training wells")
        if len(confirmation) != int(args.confirmation_wells_per_fold):
            raise RuntimeError(f"Fold {fold} has only {len(confirmation)} confirmation wells")
        training_parts.append(training)
        confirmation_parts.append(confirmation)
        fold_report.append(
            {
                "fold": int(fold),
                "training_wells": int(training["well_id"].nunique()),
                "confirmation_wells": int(confirmation["well_id"].nunique()),
            }
        )
    training = pd.concat(training_parts, ignore_index=True)
    confirmation = pd.concat(confirmation_parts, ignore_index=True)
    training_wells = set(training["well_id"])
    confirmation_wells = set(confirmation["well_id"])
    overlaps = {
        "training_design_validation": sorted(training_wells & validation_wells),
        "training_confirmation": sorted(training_wells & confirmation_wells),
        "confirmation_design_validation": sorted(confirmation_wells & validation_wells),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Stage 24 split leakage: {overlaps}")
    keep = [
        "cut_id", "well_id", "stage16_fold", "requested_fraction",
        "evaluation_role", "cut_index",
    ]
    training[keep].to_parquet(output / "training_cut_ids.parquet", index=False)
    confirmation[keep].to_parquet(output / "confirmation_cut_ids.parquet", index=False)
    summary = {
        "stage24a_manifest_complete": True,
        "training_cuts": len(training),
        "training_wells": len(training_wells),
        "confirmation_cuts": len(confirmation),
        "confirmation_wells": len(confirmation_wells),
        "design_validation_cuts": len(validation),
        "design_validation_wells": len(validation_wells),
        "fold_report": fold_report,
        "overlaps": overlaps,
        "target_fraction": float(args.target_fraction),
        "next_step": "Train soft-ordinal expected-offset TCN on training_cut_ids only.",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    print(summary, flush=True)


if __name__ == "__main__":
    main()
