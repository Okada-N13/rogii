from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.evaluation.metrics import evaluate_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a guarded PF bimodal hedge to an OOF run")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--diagnostic-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output_dir = args.artifact_dir.resolve() / args.run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(args.base_run / "oof.parquet")
    diagnostic_columns = [
        "id",
        "pf_mha_shift",
        "pf_mode_separation",
        "pf_minor_mode_mass",
    ]
    diagnostic = pd.read_parquet(args.diagnostic_run / "oof.parquet", columns=diagnostic_columns)
    predictions = base.merge(diagnostic, on="id", how="left", validate="one_to_one")
    if predictions[diagnostic_columns[1:]].isna().any().any():
        raise RuntimeError("Diagnostic PF run does not cover every base prediction row")

    overlay = config.get("overlay", {})
    active = predictions["pf_mha_shift"].ne(0.0)
    for lower, upper in overlay.get("excluded_separation_ranges", []):
        active &= ~predictions["pf_mode_separation"].between(float(lower), float(upper), inclusive="both")
    weight = float(overlay.get("weight", 0.75))
    if not 0.0 <= weight <= 1.0:
        raise ValueError("overlay.weight must be between zero and one")
    predictions["mha_active"] = active
    predictions["mha_overlay_weight"] = weight
    predictions["mha_applied_shift"] = predictions["pf_mha_shift"].where(active, 0.0) * weight
    predictions["y_pred"] = predictions["y_pred"] + predictions["mha_applied_shift"]

    metrics, well_metrics = evaluate_predictions(predictions)
    metrics["mha_active_wells"] = int(predictions.loc[active, "well_id"].nunique())
    metrics["mha_active_rows"] = int(active.sum())
    predictions.to_parquet(output_dir / "oof.parquet", index=False)
    well_metrics.to_parquet(output_dir / "per_well_metrics.parquet", index=False)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "environment.json", environment_report())
    config["resolved"] = {
        "base_run": str(args.base_run.resolve()),
        "diagnostic_run": str(args.diagnostic_run.resolve()),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "run_id": args.run_id,
    }
    write_yaml(output_dir / "config.yaml", config)
    (output_dir / "run.log").write_text(
        f"base_run={args.base_run}\ndiagnostic_run={args.diagnostic_run}\n"
        f"active_wells={metrics['mha_active_wells']}\npooled_rmse={metrics['pooled_rmse']:.8f}\n",
        encoding="utf-8",
    )
    print(f"pooled RMSE: {metrics['pooled_rmse']:.6f}")
    print(f"active MHA wells: {metrics['mha_active_wells']}")
    print(f"run artifacts: {output_dir}")


if __name__ == "__main__":
    main()
