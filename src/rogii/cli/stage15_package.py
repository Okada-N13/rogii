from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from rogii.artifacts import write_json
from rogii.config import load_config
from rogii.data.loading import discover_horizontal_wells, well_id_from_path
from rogii.models.stage15 import FAMILIES, fit_surface_bundle, save_surface_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Stage 15 fold-safe offline inference package")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage11-run", type=Path, required=True)
    parser.add_argument("--stage12b-run", type=Path, required=True)
    parser.add_argument("--stage12c-run", type=Path, required=True)
    parser.add_argument("--stage14-run", type=Path, required=True)
    parser.add_argument("--stage14b-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    stage11, stage12b, stage12c = args.stage11_run.resolve(), args.stage12b_run.resolve(), args.stage12c_run.resolve()
    stage14, stage14b = args.stage14_run.resolve(), args.stage14b_run.resolve()
    summary14b = load_config(stage14b / "gate_summary.json")
    if not summary14b.get("promoted_to_full_residual_training"):
        raise RuntimeError("Stage 14B did not authorize inference packaging")
    if summary14b.get("robust_extended_generic_spec") != "generic_w080_cap16":
        raise RuntimeError("Stage 15 requires the validated generic_w080_cap16 profile")

    output = args.artifact_dir.resolve() / args.run_id
    package = output / "package"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    package.mkdir(parents=True)

    records = pd.read_parquet(stage11 / "multicut_records.parquet")
    assignments = pd.read_parquet(stage11 / "well_folds.parquet")
    records = records.merge(
        assignments[["well_id", *FAMILIES]], on="well_id", how="left", validate="many_to_one"
    ) if not set(FAMILIES).issubset(records.columns) else records
    test_wells = [well_id_from_path(path) for path in discover_horizontal_wells(args.data_dir, "test")]
    overlap = assignments[assignments["well_id"].astype(str).isin(test_wells)].copy()
    if set(overlap["well_id"].astype(str)) != set(test_wells):
        raise RuntimeError("Every competition test well must have an OOF fold assignment")

    surface_config = dict(config["surface_model"])
    seed = int(config.get("seed", 42))
    for family in FAMILIES:
        for fold in sorted(int(value) for value in overlap[family].unique()):
            bundle = fit_surface_bundle(records, family, fold, surface_config, seed)
            save_surface_bundle(package / "surface" / f"{family}_{fold}.joblib", bundle)

    for fold in sorted(int(value) for value in overlap["fold"].unique()):
        _copy(stage12b / f"fold_{fold}.pt", package / "emission" / f"fold_{fold}.pt")
        _copy(stage14 / f"fold_generic_fold_{fold}.joblib", package / "residual" / f"fold_generic_{fold}.joblib")
        stacked = stage14 / f"fold_stacked_fold_{fold}.joblib"
        if stacked.is_file():
            _copy(stacked, package / "residual" / f"fold_stacked_{fold}.joblib")
    for family in ("spatial_fold", "typewell_fold"):
        for fold in sorted(int(value) for value in overlap[family].unique()):
            _copy(stage12c / f"{family}_{fold}.pt", package / "emission" / f"{family}_{fold}.pt")

    source = Path(__file__).resolve().parents[2]
    shutil.copytree(source / "rogii", package / "src" / "rogii", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    assignment_rows = overlap[["well_id", *FAMILIES]].sort_values("well_id").to_dict("records")
    manifest = {
        "stage": "15_fold_safe_independent_inference",
        "test_well_assignments": assignment_rows,
        "same_well_target_leakage_guard": True,
        "surface": {"weight": 0.75, "correction_cap_ft": 50.0},
        "ncc": dict(config["ncc"]),
        "primary_residual": {"branch": "generic", "weight": 0.80, "cap_ft": 16.0},
        "secondary_residual": {"branch": "stacked", "weight": 1.0, "cap_ft": 16.0},
    }
    write_json(package / "manifest.json", manifest)
    files = sorted(path for path in package.rglob("*") if path.is_file())
    integrity = {str(path.relative_to(package)).replace("\\", "/"): _sha256(path) for path in files}
    write_json(package / "sha256.json", integrity)
    shutil.make_archive(str(output / "stage15_inference_package"), "zip", package)
    report = {
        "package_ready": True,
        "test_wells": test_wells,
        "fold_safe_assignments": assignment_rows,
        "files": len(integrity),
        "zip": str(output / "stage15_inference_package.zip"),
        "same_well_target_leakage_guard": True,
    }
    write_json(output / "package_summary.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
