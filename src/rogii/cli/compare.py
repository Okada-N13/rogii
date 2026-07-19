from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.artifacts import write_json
from rogii.evaluation.metrics import paired_well_bootstrap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare ROGII experiment run directories")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _read_metrics(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    baseline_oof = pd.read_parquet(args.baseline / "oof.parquet").sort_values("id").reset_index(drop=True)
    baseline_wells = pd.read_parquet(args.baseline / "per_well_metrics.parquet").set_index("well_id")
    baseline_metrics = _read_metrics(args.baseline)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    delta_frames: list[pd.DataFrame] = []

    for candidate_dir in args.candidate:
        candidate_oof = pd.read_parquet(candidate_dir / "oof.parquet").sort_values("id").reset_index(drop=True)
        if not baseline_oof["id"].equals(candidate_oof["id"]):
            raise ValueError(f"OOF rows do not align: {candidate_dir}")
        candidate_wells = pd.read_parquet(candidate_dir / "per_well_metrics.parquet").set_index("well_id")
        if not baseline_wells.index.equals(candidate_wells.index):
            candidate_wells = candidate_wells.reindex(baseline_wells.index)
        candidate_metrics = _read_metrics(candidate_dir)
        delta = candidate_wells["rmse"] - baseline_wells["rmse"]
        baseline_error = baseline_oof["y_pred"].to_numpy() - baseline_oof["y_true"].to_numpy()
        candidate_error = candidate_oof["y_pred"].to_numpy() - candidate_oof["y_true"].to_numpy()
        bootstrap = paired_well_bootstrap(
            candidate_oof,
            baseline_oof,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
        name = candidate_dir.name
        records.append(
            {
                "run": name,
                "pooled_rmse": candidate_metrics["pooled_rmse"],
                "pooled_rmse_delta": float(candidate_metrics["pooled_rmse"]) - float(baseline_metrics["pooled_rmse"]),
                "well_rmse_median": candidate_metrics["well_rmse_median"],
                "well_rmse_p90": candidate_metrics["well_rmse_p90"],
                "worst_10pct_sse_share": candidate_metrics["worst_10pct_sse_share"],
                "improved_wells": int((delta < 0).sum()),
                "worsened_wells": int((delta > 0).sum()),
                "unchanged_wells": int((delta == 0).sum()),
                "improved_fraction": float((delta < 0).mean()),
                "mean_well_rmse_delta": float(delta.mean()),
                "max_well_improvement": float(-delta.min()),
                "max_well_degradation": float(delta.max()),
                "error_correlation_with_baseline": float(np.corrcoef(candidate_error, baseline_error)[0, 1]),
                "bootstrap_ci_2_5": bootstrap["ci_2_5"],
                "bootstrap_ci_97_5": bootstrap["ci_97_5"],
            }
        )
        delta_frames.append(
            pd.DataFrame(
                {
                    "run": name,
                    "well_id": baseline_wells.index,
                    "baseline_rmse": baseline_wells["rmse"].to_numpy(),
                    "candidate_rmse": candidate_wells["rmse"].to_numpy(),
                    "rmse_delta": delta.to_numpy(),
                }
            )
        )

    comparison = pd.DataFrame.from_records(records).sort_values("pooled_rmse")
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    pd.concat(delta_frames, ignore_index=True).to_parquet(output_dir / "well_deltas.parquet", index=False)
    write_json(
        output_dir / "comparison.json",
        {"baseline": args.baseline.name, "candidates": comparison.to_dict(orient="records")},
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

