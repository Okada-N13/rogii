from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json
from rogii.models.portable_hgb import export_hist_gradient_boosting, load_portable_hgb, predict_portable_hgb


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the portable Stage 19C Internet-OFF inference package")
    parser.add_argument("--stage19b-run", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source_run = args.stage19b_run.resolve()
    source_summary = json.loads((source_run / "summary.json").read_text(encoding="utf-8"))
    if not source_summary.get("promoted_to_stage19c"):
        raise AssertionError("Stage 19B was not promoted to Stage 19C")
    source = source_run / "stage19b_trajectory_bundle"
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    package = output / "stage19c_trajectory_inference_package"
    models_dir = package / "models"
    models_dir.mkdir(parents=True)
    portable_report = []
    parity_max = 0.0
    rng = np.random.default_rng(19)
    probe = rng.normal(size=(32, len(source_manifest["feature_columns"])))
    probe[::7, ::11] = np.nan
    for item in source_manifest["models"]:
        source_path = source / item["file"]
        with source_path.open("rb") as handle:
            model = pickle.load(handle)
        target_path = models_dir / (source_path.stem + ".npz")
        export_hist_gradient_boosting(model, target_path)
        portable = load_portable_hgb(target_path)
        difference = float(np.max(np.abs(model.predict(pd.DataFrame(probe, columns=source_manifest["feature_columns"])) - predict_portable_hgb(portable, probe))))
        parity_max = max(parity_max, difference)
        portable_report.append({
            "seed": int(item["seed"]), "target": str(item["target"]),
            "file": f"models/{target_path.name}", "bytes": target_path.stat().st_size,
            "sha256": _sha256(target_path),
        })
    inference_source = Path(__file__).resolve().parents[1] / "inference" / "stage19_trajectory.py"
    shutil.copy2(inference_source, package / "stage19_trajectory.py")
    manifest = {
        "stage19c_trajectory_inference_package": True, "package_version": 2,
        "source_stage19b_manifest_sha256": source_summary["package_manifest_sha256"],
        "feature_columns": source_manifest["feature_columns"],
        "coefficient_columns": source_manifest["coefficient_columns"],
        "features": source_manifest["features"], "profile": source_manifest["profile"],
        "models": portable_report, "inference_file": "stage19_trajectory.py",
        "inference_sha256": _sha256(package / "stage19_trajectory.py"),
        "runtime_contract": {
            "internet": False, "gpu": False, "sklearn_required": False,
            "hidden_target_columns_used": False, "outputs_per_well": 3,
        },
    }
    write_json(package / "manifest.json", manifest)
    archive = output / "stage19c_trajectory_inference_package.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(package))
    gates = {
        "stage19b_promoted": True, "portable_model_count_matches": len(portable_report) == len(source_manifest["models"]),
        "portable_prediction_exact": parity_max <= 1e-12, "inference_module_packaged": (package / "stage19_trajectory.py").is_file(),
        "no_pickle_models": not any(package.rglob("*.pkl")),
    }
    summary = {
        "stage19c_package_complete": True, "package_ready": bool(all(gates.values())),
        "portable_models": len(portable_report), "portable_prediction_max_abs_difference": parity_max,
        "package_manifest_sha256": _sha256(package / "manifest.json"), "zip": str(archive),
        "gates": gates,
        "next_step": "Upload the zip as a Kaggle Dataset and run notebook 510 with Internet OFF.",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    print(summary, flush=True)


if __name__ == "__main__":
    main()
