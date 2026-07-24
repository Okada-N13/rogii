from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii.artifacts import write_json
from rogii.inference.stage18_retrieval import apply_ranked_retrieval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit exact Stage 18 serial/parallel inference parity")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-wells", type=int, default=200)
    parser.add_argument("--runtime-gate-seconds", type=float, default=600.0)
    return parser


def projected_hidden_runtime(
    serial_seconds: float,
    parallel_seconds: float,
    public_wells: int,
    hidden_wells: int,
    workers: int,
) -> float:
    """Conservative projection that retains the observed parallel run as fixed overhead."""
    if min(serial_seconds, parallel_seconds, public_wells, hidden_wells, workers) <= 0:
        raise ValueError("Runtime projection inputs must be positive")
    remaining_batches = math.ceil(max(0, hidden_wells - public_wells) / workers)
    conservative_per_well = serial_seconds / public_wells
    return float(parallel_seconds + remaining_batches * conservative_per_well)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_submission(path: Path, sample: pd.DataFrame) -> dict[str, Any]:
    frame = pd.read_csv(path)
    return {
        "rows": int(len(frame)),
        "id_order_matches_sample": bool(frame["id"].astype(str).equals(sample["id"].astype(str))),
        "finite_tvt": bool(np.isfinite(frame["tvt"]).all()),
        "sha256": _sha256(path),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    package_dir, data_dir = args.package_dir.resolve(), args.data_dir.resolve()
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty benchmark: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("package_version", 0)) < 4:
        raise AssertionError("Stage 36 requires Stage 18 package version 4 or newer")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if list(sample.columns) != ["id", "tvt"] or not np.isfinite(sample["tvt"]).all():
        raise AssertionError("The competition sample must be a finite id/tvt benchmark input")

    serial_path, parallel_path = output / "serial_submission.csv", output / "parallel_submission.csv"
    shutil.copy2(data_dir / "sample_submission.csv", serial_path)
    serial = apply_ranked_retrieval(package_dir, data_dir, serial_path, well_workers=1)
    shutil.copy2(data_dir / "sample_submission.csv", parallel_path)
    parallel = apply_ranked_retrieval(package_dir, data_dir, parallel_path, well_workers=args.workers)

    serial_audit, parallel_audit = _audit_submission(serial_path, sample), _audit_submission(parallel_path, sample)
    exact_bytes = serial_path.read_bytes() == parallel_path.read_bytes()
    public_wells = int(sample["id"].astype(str).str.rsplit("_", n=1).str[0].nunique())
    projection = projected_hidden_runtime(
        float(serial["elapsed_seconds"]),
        float(parallel["elapsed_seconds"]),
        public_wells,
        int(args.hidden_wells),
        int(args.workers),
    )
    gates = {
        "exact_submission_bytes": exact_bytes,
        "serial_output_valid": serial_audit["id_order_matches_sample"] and serial_audit["finite_tvt"],
        "parallel_output_valid": parallel_audit["id_order_matches_sample"] and parallel_audit["finite_tvt"],
        "all_public_wells_audited": len(serial["well_report"]) == public_wells == len(parallel["well_report"]),
        "hidden_runtime_projection_below_gate": projection <= float(args.runtime_gate_seconds),
    }
    promoted = all(gates.values())
    summary = {
        "stage36_complete": True,
        "promoted_to_kaggle_package_v4": promoted,
        "package_version": int(manifest["package_version"]),
        "public_wells": public_wells,
        "hidden_wells_projection": int(args.hidden_wells),
        "workers": int(args.workers),
        "serial_elapsed_seconds": float(serial["elapsed_seconds"]),
        "parallel_elapsed_seconds": float(parallel["elapsed_seconds"]),
        "observed_speedup": float(serial["elapsed_seconds"] / parallel["elapsed_seconds"]),
        "projected_hidden_elapsed_seconds": projection,
        "runtime_gate_seconds": float(args.runtime_gate_seconds),
        "serial_submission": serial_audit,
        "parallel_submission": parallel_audit,
        "gates": gates,
        "next_step": (
            "Upload the v4 package and run the Kaggle Stage 18 notebook."
            if promoted
            else "Do not submit; optimize Stage 18 runtime while preserving exact output bytes."
        ),
    }
    write_json(output / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
