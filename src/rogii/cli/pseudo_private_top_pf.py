from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.selector_replay import likelihood_selector
from rogii.cli.trajectory_base_alignment import top_pf_proxy
from rogii.config import load_config


MATERIALIZED_ROLES = ("training", "design_validation")
FOLD_COLUMNS = ("fold", "spatial_fold", "typewell_fold", "branch_group_fold")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the Stage 37 top-PF pseudo-private base")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--stage37a-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit-cuts", type=int)
    return parser


def _rmse(sse: float, rows: int) -> float:
    return float(np.sqrt(float(sse) / int(rows))) if rows else float("nan")


def _aggregate(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    rows = int(frame["suffix_rows"].sum())
    report: dict[str, Any] = {
        "cuts": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "rows": rows,
        "public_rmse": _rmse(frame["public_sse"].sum(), rows),
        "selector_rmse": _rmse(frame["selector_sse"].sum(), rows),
        "top_pf_proxy_rmse": _rmse(frame["top_pf_proxy_sse"].sum(), rows),
    }
    for column in columns:
        groups = {}
        for value, group in frame.groupby(column, sort=True):
            group_rows = int(group["suffix_rows"].sum())
            groups[str(value)] = {
                "cuts": int(len(group)),
                "rows": group_rows,
                "top_pf_proxy_rmse": _rmse(group["top_pf_proxy_sse"].sum(), group_rows),
            }
        report[f"{column}_report"] = groups
    return report


def _chunk_paths(chunks: Path, index: int) -> tuple[Path, Path]:
    stem = f"chunk_{index:04d}"
    return chunks / f"{stem}_predictions.parquet", chunks / f"{stem}_cuts.parquet"


def _validate_existing_chunk(
    prediction_path: Path,
    metric_path: Path,
    expected_ids: list[str],
) -> bool:
    if not prediction_path.is_file() or not metric_path.is_file():
        return False
    metrics = pd.read_parquet(metric_path, columns=["cut_id"])
    actual = metrics["cut_id"].astype(str).tolist()
    if actual != expected_ids:
        raise AssertionError(f"Resume chunk IDs changed: {metric_path}")
    predictions = pd.read_parquet(prediction_path, columns=["cut_id", "top_pf_proxy_y_pred"])
    if set(predictions["cut_id"].astype(str)) != set(expected_ids):
        raise AssertionError(f"Resume prediction coverage changed: {prediction_path}")
    if not np.isfinite(predictions["top_pf_proxy_y_pred"]).all():
        raise AssertionError(f"Resume chunk contains non-finite predictions: {prediction_path}")
    return True


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    stage17 = args.stage17a_run.resolve()
    stage37 = args.stage37a_run.resolve()
    data_dir = args.data_dir.resolve()
    output = args.artifact_dir.resolve() / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    chunks = output / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)

    source_summary = json.loads((stage37 / "summary.json").read_text(encoding="utf-8"))
    expected_manifest = str(config["provenance"]["stage37a_manifest_sha256"])
    if source_summary.get("pseudo_private_manifest_sha256") != expected_manifest:
        raise AssertionError("Stage 37A manifest hash changed")
    if not source_summary.get("promoted_to_stage37b_top_pf_replay"):
        raise AssertionError("Stage 37A did not promote")
    manifest = pd.read_parquet(stage37 / "pseudo_private_manifest.parquet")
    confirmation = manifest[manifest["benchmark_role"] == "confirmation_locked"]
    selected = manifest[manifest["benchmark_role"].isin(MATERIALIZED_ROLES)].copy()
    selected = selected.sort_values(["benchmark_role", "well_id", "cut_index", "cut_id"], kind="stable")
    if args.limit_cuts is not None:
        selected = selected.head(int(args.limit_cuts)).copy()
    selected = selected.reset_index(drop=True)
    if selected.empty:
        raise AssertionError("Stage 37B has no selected cuts")
    if set(selected["well_id"]).intersection(set(confirmation["well_id"])):
        raise AssertionError("Confirmation wells entered Stage 37B materialization")

    cut_ids = selected["cut_id"].astype(str).tolist()
    public = pd.read_parquet(
        stage17 / "replay_predictions.parquet",
        columns=["cut_id", "row_index", "y_pred"],
        filters=[("cut_id", "in", cut_ids)],
    )
    public["cut_id"] = public["cut_id"].astype(str)
    if set(public["cut_id"].unique()) != set(cut_ids):
        missing = sorted(set(cut_ids) - set(public["cut_id"].unique()))
        raise AssertionError(f"Public replay does not cover Stage 37B cuts: {missing[:5]}")
    public_groups = {
        cut_id: frame.sort_values("row_index")
        for cut_id, frame in public.groupby("cut_id", sort=False)
    }
    train_dir = data_dir / "train"

    @lru_cache(maxsize=64)
    def load_horizontal(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train_dir / f"{well_id}__horizontal_well.csv")

    @lru_cache(maxsize=64)
    def load_typewell(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train_dir / f"{well_id}__typewell.csv")

    selector_config = dict(config.get("selector", {}))
    proxy_config = dict(config.get("top_pf_proxy", {}))
    chunk_size = int(config.get("storage", {}).get("cuts_per_chunk", 20))
    if chunk_size <= 0:
        raise ValueError("cuts_per_chunk must be positive")
    chunk_report = []
    invariance_checks: list[bool] = []
    total_chunks = int(np.ceil(len(selected) / chunk_size))
    for chunk_index, start in enumerate(range(0, len(selected), chunk_size)):
        batch = selected.iloc[start : start + chunk_size].copy()
        expected_ids = batch["cut_id"].astype(str).tolist()
        prediction_path, metric_path = _chunk_paths(chunks, chunk_index)
        if _validate_existing_chunk(prediction_path, metric_path, expected_ids):
            chunk_report.append({
                "chunk": chunk_index, "cuts": len(batch), "status": "reused",
                "prediction_file": prediction_path.name, "metric_file": metric_path.name,
            })
            print(f"top-PF chunk {chunk_index + 1}/{total_chunks} reused", flush=True)
            continue
        if prediction_path.exists():
            prediction_path.unlink()
        if metric_path.exists():
            metric_path.unlink()
        prediction_parts, metric_rows = [], []
        for cut in batch.itertuples(index=False):
            cut_id, well_id, cut_index = str(cut.cut_id), str(cut.well_id), int(cut.cut_index)
            horizontal, typewell = load_horizontal(well_id), load_typewell(well_id)
            public_frame = public_groups[cut_id]
            row_index = public_frame["row_index"].to_numpy(np.int64)
            expected_rows = np.arange(cut_index, len(horizontal), dtype=np.int64)
            if not np.array_equal(row_index, expected_rows):
                raise AssertionError(f"{cut_id}: public replay row coverage mismatch")
            public_prediction = public_frame["y_pred"].to_numpy(float)
            masked = horizontal[["MD", "Z", "GR", "TVT"]].copy()
            masked.loc[masked.index >= cut_index, "TVT"] = np.nan
            selector, selector_audit = likelihood_selector(masked, typewell, cut_index, selector_config)
            proxy = top_pf_proxy(horizontal, cut_index, public_prediction, selector, proxy_config)
            truth = horizontal["TVT"].to_numpy(float)[cut_index:]
            hidden_target_invariant: bool | None = None
            if len(invariance_checks) < int(config.get("audit", {}).get("invariance_cuts", 12)):
                changed = horizontal.copy()
                changed.loc[changed.index >= cut_index, "TVT"] += 1_000_000.0
                masked_changed = changed[["MD", "Z", "GR", "TVT"]].copy()
                masked_changed.loc[masked_changed.index >= cut_index, "TVT"] = np.nan
                selector_changed, _ = likelihood_selector(
                    masked_changed, typewell, cut_index, selector_config
                )
                proxy_changed = top_pf_proxy(
                    changed, cut_index, public_prediction, selector_changed, proxy_config
                )
                hidden_target_invariant = bool(
                    np.array_equal(selector, selector_changed) and np.array_equal(proxy, proxy_changed)
                )
                invariance_checks.append(hidden_target_invariant)
            prediction_parts.append(pd.DataFrame({
                "cut_id": cut_id,
                "well_id": well_id,
                "benchmark_role": str(cut.benchmark_role),
                "row_index": expected_rows.astype(np.int32),
                "y_true": truth,
                "public_y_pred": public_prediction,
                "selector_y_pred": selector,
                "top_pf_proxy_y_pred": proxy,
            }))
            metric_rows.append({
                "cut_id": cut_id,
                "well_id": well_id,
                "benchmark_role": str(cut.benchmark_role),
                "requested_fraction": float(cut.requested_fraction),
                "suffix_rows": len(truth),
                **{column: int(getattr(cut, column)) for column in FOLD_COLUMNS},
                "public_sse": float(np.square(public_prediction - truth).sum()),
                "selector_sse": float(np.square(selector - truth).sum()),
                "top_pf_proxy_sse": float(np.square(proxy - truth).sum()),
                "hidden_target_invariant": hidden_target_invariant,
                **selector_audit,
            })
        predictions = pd.concat(prediction_parts, ignore_index=True)
        metrics = pd.DataFrame.from_records(metric_rows)
        predictions.to_parquet(prediction_path, index=False, compression="zstd")
        metrics.to_parquet(metric_path, index=False)
        chunk_report.append({
            "chunk": chunk_index, "cuts": len(batch), "status": "computed",
            "prediction_file": prediction_path.name, "metric_file": metric_path.name,
        })
        print(f"top-PF chunk {chunk_index + 1}/{total_chunks} computed", flush=True)

    metric_files = [_chunk_paths(chunks, index)[1] for index in range(total_chunks)]
    prediction_files = [_chunk_paths(chunks, index)[0] for index in range(total_chunks)]
    metrics = pd.concat([pd.read_parquet(path) for path in metric_files], ignore_index=True)
    if metrics["cut_id"].astype(str).tolist() != cut_ids:
        raise AssertionError("Stage 37B final metric order changed")
    prediction_rows = sum(
        len(pd.read_parquet(path, columns=["cut_id"])) for path in prediction_files
    )
    expected_rows = int(selected["suffix_rows"].sum())
    if prediction_rows != expected_rows:
        raise AssertionError(f"Stage 37B prediction rows {prediction_rows} != {expected_rows}")
    metrics.to_parquet(output / "cut_metrics.parquet", index=False)
    persisted_invariance = metrics["hidden_target_invariant"].dropna().astype(bool).tolist()
    write_json(output / "prediction_index.json", {
        "format": "stage37b_chunked_parquet_v1",
        "manifest_sha256": expected_manifest,
        "prediction_files": [str(path.relative_to(output)) for path in prediction_files],
        "rows": prediction_rows,
    })
    role_reports = {
        role: _aggregate(
            metrics[metrics["benchmark_role"] == role],
            ["requested_fraction", *FOLD_COLUMNS],
        )
        for role in MATERIALIZED_ROLES
    }
    confirmation_wells_materialized = sorted(
        set(metrics["well_id"]).intersection(set(confirmation["well_id"]))
    )
    gates = {
        "stage37a_manifest_matches": True,
        "training_and_design_only": not confirmation_wells_materialized,
        "confirmation_target_unread": not confirmation_wells_materialized,
        "prediction_rows_complete": prediction_rows == expected_rows,
        "prediction_chunks_complete": all(path.is_file() for path in prediction_files),
        "hidden_target_invariance": bool(persisted_invariance) and all(persisted_invariance),
        "finite_role_metrics": all(
            np.isfinite(report["top_pf_proxy_rmse"]) for report in role_reports.values()
        ),
    }
    summary = {
        "stage37b_complete": True,
        "promoted_to_stage38_retrieval_v2": bool(all(gates.values())),
        "stage37a_manifest_sha256": expected_manifest,
        "materialized_cuts": len(selected),
        "materialized_wells": int(selected["well_id"].nunique()),
        "prediction_rows": prediction_rows,
        "chunks": total_chunks,
        "chunk_report": chunk_report,
        "roles": role_reports,
        "confirmation_wells_materialized": confirmation_wells_materialized,
        "confirmation_target_columns_read": False,
        "proxy_limitations": [
            "fold-safe public replay substitutes for unavailable complete V599 learned branch state",
            "visible-prefix adaptive profile, model-package overlay, and seed-branch hedge are omitted",
            "the proxy is a stable model-development control, not an absolute public-LB estimator",
        ],
        "gates": gates,
        "next_step": (
            "Train Stage 38 donor-retrieval v2 on training chunks and evaluate once on design validation."
            if all(gates.values())
            else "Repair Stage 37B materialization before training a new model."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_report())
    resolved = dict(config)
    resolved["resolved"] = {
        "stage17a_run": str(stage17),
        "stage37a_run": str(stage37),
        "data_dir": str(data_dir),
        "run_id": args.run_id,
        "limit_cuts": args.limit_cuts,
    }
    write_yaml(output / "config.yaml", resolved)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
