from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rogii.artifacts import environment_report, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expand frozen Stage 24 wells to all safe primary cuts")
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--base-manifest-run", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stage17 = args.stage17a_run.resolve()
    base_manifest = args.base_manifest_run.resolve()
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    base_summary = json.loads((base_manifest / "summary.json").read_text(encoding="utf-8"))
    if base_summary.get("training_wells") != 500 or base_summary.get("confirmation_wells") != 120:
        raise AssertionError("Base manifest is not the frozen 500/120 split")
    base_training = pd.read_parquet(base_manifest / "training_cut_ids.parquet")
    confirmation = pd.read_parquet(base_manifest / "confirmation_cut_ids.parquet")
    training_wells = set(base_training["well_id"].astype(str))
    confirmation_wells = set(confirmation["well_id"].astype(str))
    if training_wells & confirmation_wells:
        raise AssertionError("Base manifest training/confirmation overlap")
    cuts = pd.read_parquet(stage17 / "cut_report.parquet")
    expanded = cuts[
        cuts["well_id"].astype(str).isin(training_wells)
        & (cuts["evaluation_role"] == "primary")
        & cuts["replay_eligible"].astype(bool)
    ].copy()
    expanded["well_id"] = expanded["well_id"].astype(str)
    if set(expanded["well_id"]) != training_wells:
        missing = sorted(training_wells - set(expanded["well_id"]))
        raise AssertionError(f"Training wells without safe primary cuts: {missing[:5]}")
    if set(expanded["well_id"]) & confirmation_wells:
        raise AssertionError("Expanded cuts opened confirmation wells")
    keep = [
        "cut_id", "well_id", "stage16_fold", "requested_fraction",
        "evaluation_role", "cut_index",
    ]
    expanded.sort_values(["well_id", "cut_index"])[keep].to_parquet(
        output / "training_cut_ids.parquet", index=False
    )
    confirmation.to_parquet(output / "confirmation_cut_ids.parquet", index=False)
    fraction_counts = {
        str(float(key)): int(value)
        for key, value in expanded["requested_fraction"].value_counts().sort_index().items()
    }
    summary = {
        "stage29a_manifest_complete": True,
        "public_replay_eligible_only": True,
        "training_cuts": len(expanded),
        "training_wells": len(training_wells),
        "confirmation_cuts": len(confirmation),
        "confirmation_wells": len(confirmation_wells),
        "training_fraction_counts": fraction_counts,
        "overlaps": base_summary.get("overlaps", {}),
        "base_manifest_run": str(base_manifest),
        "reserved_confirmation_used": False,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    print(summary, flush=True)


if __name__ == "__main__":
    main()

