from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json
from rogii.cli.donor_ranker import FEATURE_COLUMNS, _model, _training_mask
from rogii.inference.stage18_retrieval import export_hist_gradient_boosting


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage 18 fold-safe test inference package")
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage18d-run", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stage16, stage18d = args.stage16b_run.resolve(), args.stage18d_run.resolve()
    stage18d_summary = json.loads((stage18d / "summary.json").read_text(encoding="utf-8"))
    if not stage18d_summary.get("promoted_to_full_ranker_training"):
        raise AssertionError("Stage 18D ranker was not promoted")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty package: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows = pd.read_parquet(stage18d / "donor_training_rows.parquet")
    config_path = stage18d / "config.yaml"
    import yaml
    stage18d_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ranker_config = dict(stage18d_config.get("ranker", {}))
    model_report, model_hashes = [], {}
    for fold in sorted(rows["target_branch_fold"].unique()):
        mask = _training_mask(rows, int(fold))
        train = rows.loc[mask]
        model = _model({**ranker_config, "seed": int(stage18d_config.get("seed", 42)) + int(fold)})
        model.fit(
            train[FEATURE_COLUMNS], train["label_log_rmse"],
            sample_weight=np.sqrt(train["suffix_rows"].to_numpy(float)),
        )
        model_path = output / f"ranker_fold_{int(fold)}.json"
        model_path.write_text(json.dumps(export_hist_gradient_boosting(model), separators=(",", ":")), encoding="utf-8")
        model_hashes[model_path.name] = _sha256(model_path)
        model_report.append({"fold": int(fold), "training_rows": len(train), "sha256": model_hashes[model_path.name]})

    assignments_source = stage16 / "well_assignments.parquet"
    assignments = pd.read_parquet(assignments_source)[["well_id", "branch_group", "branch_group_fold"]]
    assignments.to_parquet(output / "well_assignments.parquet", index=False)
    inference_source = Path(__file__).resolve().parents[1] / "inference" / "stage18_retrieval.py"
    shutil.copy2(inference_source, output / "stage18_retrieval.py")
    manifest = {
        "stage18e_package": True, "stage18d_promoted": True,
        "stage16b_manifest_sha256": stage18d_summary["stage16b_manifest_sha256"],
        "stage18c_sample_sha256": stage18d_summary["stage18c_sample_sha256"],
        "feature_columns": FEATURE_COLUMNS, "model_report": model_report,
        "same_well_target_transfer_removed": True,
        "inference": {
            "n_folds": 5, "trajectory_samples": 48, "query_prefix_samples": 24, "point_neighbors": 64,
            "donor_neighbors": 16, "maximum_candidates": 12, "minimum_donors": 2, "selected_donors": 4,
            "branch_distance_ft": 150.0, "minimum_matched_points": 3,
            "prefix_calibration_rows": 256, "distance_scale_ft": 300.0, "gr_scale": 30.0,
            "minimum_weight": 1.0e-5, "blend_weight": 0.20,
        },
        "files": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file():
            manifest["files"][path.name] = _sha256(path)
    write_json(output / "manifest.json", manifest)
    write_json(output / "environment.json", environment_report())
    archive = output / "stage18e_ranked_retrieval_package.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.iterdir()):
            if path.is_file() and path != archive:
                bundle.write(path, path.name)
    summary = {
        "stage18e_package_complete": True, "fold_models": len(model_report),
        "training_candidate_rows": len(rows), "same_well_target_leakage_guard": True,
        "package_manifest_sha256": _sha256(output / "manifest.json"), "zip": str(archive),
        "next_step": "Upload the zip as a Kaggle Dataset and run the V599 Stage 18 postprocess notebook.",
    }
    write_json(output / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
