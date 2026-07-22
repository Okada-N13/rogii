from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config


REQUIRED_PUBLIC_COLUMNS = {
    "id", "well_id", "row_index", "fold", "y_true", "y_pred",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay provenance-safe public OOF on Stage 16B cuts")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--public-oof-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit-wells", type=int)
    return parser


def replay_is_target_safe(original_cut: int, pseudo_cut: int) -> bool:
    """The frozen prediction may only use an original prefix contained in the pseudo prefix."""
    return int(original_cut) <= int(pseudo_cut)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rmse(sse: float, rows: int) -> float:
    return float(np.sqrt(float(sse) / int(rows))) if rows else float("nan")


def _summarize_cuts(cuts: pd.DataFrame, role: str) -> dict[str, Any]:
    selected = cuts[cuts["evaluation_role"] == role]
    eligible = selected[selected["replay_eligible"]]
    baseline_sse = float(selected["baseline_sse"].sum())
    hybrid_sse = float(selected["hybrid_sse"].sum())
    rows = int(selected["suffix_rows"].sum())
    valid_rows = int(eligible["suffix_rows"].sum())
    valid_baseline_sse = float(eligible["baseline_sse"].sum())
    public_sse = float(eligible["public_sse"].sum())
    return {
        "cuts": int(len(selected)),
        "eligible_cuts": int(len(eligible)),
        "cut_coverage": float(len(eligible) / len(selected)) if len(selected) else 0.0,
        "rows": rows,
        "eligible_rows": valid_rows,
        "row_coverage": float(valid_rows / rows) if rows else 0.0,
        "baseline_rmse": _rmse(baseline_sse, rows),
        "hybrid_rmse": _rmse(hybrid_sse, rows),
        "hybrid_delta": _rmse(hybrid_sse, rows) - _rmse(baseline_sse, rows),
        "eligible_baseline_rmse": _rmse(valid_baseline_sse, valid_rows),
        "eligible_public_rmse": _rmse(public_sse, valid_rows),
        "eligible_public_delta": _rmse(public_sse, valid_rows) - _rmse(valid_baseline_sse, valid_rows),
    }


def _fold_report(cuts: pd.DataFrame, role: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for fold, selected in cuts[cuts["evaluation_role"] == role].groupby("stage16_fold", sort=True):
        eligible = selected[selected["replay_eligible"]]
        rows, valid_rows = int(selected["suffix_rows"].sum()), int(eligible["suffix_rows"].sum())
        base = _rmse(float(selected["baseline_sse"].sum()), rows)
        hybrid = _rmse(float(selected["hybrid_sse"].sum()), rows)
        valid_base = _rmse(float(eligible["baseline_sse"].sum()), valid_rows)
        public = _rmse(float(eligible["public_sse"].sum()), valid_rows)
        output.append({
            "fold": int(fold), "rows": rows, "eligible_rows": valid_rows,
            "row_coverage": float(valid_rows / rows) if rows else 0.0,
            "baseline_rmse": base, "hybrid_rmse": hybrid, "hybrid_delta": hybrid - base,
            "eligible_baseline_rmse": valid_base, "eligible_public_rmse": public,
            "eligible_public_delta": public - valid_base,
        })
    return output


def _write_prediction(writer: pq.ParquetWriter | None, path: Path, frame: pd.DataFrame) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    stage16_run, public_run = args.stage16b_run.resolve(), args.public_oof_run.resolve()
    manifest_path, stage16_summary_path = stage16_run / "pseudo_test_manifest.parquet", stage16_run / "summary.json"
    public_path = public_run / "base_oof.parquet"
    for path in [manifest_path, stage16_summary_path, public_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    stage16_summary = json.loads(stage16_summary_path.read_text(encoding="utf-8"))
    expected_manifest = str(config["provenance"]["stage16b_manifest_sha256"])
    if str(stage16_summary.get("manifest_sha256")) != expected_manifest:
        raise AssertionError("Stage 16B manifest hash does not match the frozen config")

    manifest = pd.read_parquet(manifest_path).sort_values(["well_id", "cut_index"], kind="stable")
    public = pd.read_parquet(public_path)
    missing = sorted(REQUIRED_PUBLIC_COLUMNS - set(public.columns))
    if missing:
        raise ValueError(f"Public base_oof is missing columns: {missing}")
    public = public[sorted(REQUIRED_PUBLIC_COLUMNS)].copy()
    public["well_id"] = public["well_id"].astype(str)
    public["row_index"] = pd.to_numeric(public["row_index"], errors="raise").astype(np.int32)
    if public["id"].duplicated().any():
        raise AssertionError("Public base_oof contains duplicate IDs")
    fold_counts = public.groupby("well_id", sort=True)["fold"].nunique()
    if int(fold_counts.max()) != 1 or int(public["fold"].nunique()) != 5:
        raise AssertionError("Public OOF fold provenance is not well-isolated 5-fold data")

    selected_wells = sorted(manifest["well_id"].astype(str).unique())
    if args.limit_wells is not None:
        selected_wells = selected_wells[: int(args.limit_wells)]
        manifest = manifest[manifest["well_id"].astype(str).isin(selected_wells)]
        public = public[public["well_id"].isin(selected_wells)]

    public_by_well = {well: frame.sort_values("row_index", kind="stable") for well, frame in public.groupby("well_id", sort=True)}
    cut_rows: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    prediction_path = output / "replay_predictions.parquet"
    maximum_target_difference = 0.0

    try:
        for position, well_id in enumerate(selected_wells, 1):
            if well_id not in public_by_well:
                raise AssertionError(f"Public OOF has no rows for well {well_id}")
            source = public_by_well[well_id]
            original_cut = int(source["row_index"].min())
            horizontal = pd.read_csv(
                args.data_dir.resolve() / "train" / f"{well_id}__horizontal_well.csv",
                usecols=["MD", "TVT"],
            )
            source_index = source["row_index"].to_numpy(dtype=np.int64)
            expected_index = np.arange(original_cut, len(horizontal), dtype=np.int64)
            if not np.array_equal(source_index, expected_index):
                raise AssertionError(f"{well_id}: public OOF is not a contiguous suffix")
            raw_source_truth = horizontal["TVT"].to_numpy(dtype=float)[source_index]
            difference = float(np.max(np.abs(raw_source_truth - source["y_true"].to_numpy(dtype=float))))
            maximum_target_difference = max(maximum_target_difference, difference)
            if difference > float(config["provenance"].get("target_tolerance", 1e-5)):
                raise AssertionError(f"{well_id}: public OOF target alignment differs by {difference}")

            public_fold = int(source["fold"].iloc[0])
            for cut in manifest[manifest["well_id"].astype(str) == well_id].itertuples(index=False):
                cut_index = int(cut.cut_index)
                truth = horizontal["TVT"].to_numpy(dtype=float)[cut_index:]
                md = horizontal["MD"].to_numpy(dtype=float)[cut_index:]
                baseline = np.full(len(truth), float(horizontal["TVT"].iloc[cut_index - 1]), dtype=float)
                baseline_sse = float(np.square(baseline - truth).sum())
                eligible = replay_is_target_safe(original_cut, cut_index)
                public_sse = float("nan")
                hybrid_sse = baseline_sse
                if eligible:
                    offset = cut_index - original_cut
                    prediction = source["y_pred"].to_numpy(dtype=float)[offset:]
                    if len(prediction) != len(truth):
                        raise AssertionError(f"{cut.cut_id}: public replay length mismatch")
                    public_sse = float(np.square(prediction - truth).sum())
                    hybrid_sse = public_sse
                    prediction_frame = pd.DataFrame({
                        "cut_id": str(cut.cut_id), "source_well_id": well_id,
                        "row_index": np.arange(cut_index, len(horizontal), dtype=np.int32),
                        "MD": md, "stage16_fold": np.int16(cut.fold), "public_fold": np.int16(public_fold),
                        "evaluation_role": str(cut.evaluation_role),
                        "requested_fraction": np.float32(cut.requested_fraction),
                        "y_true": truth, "y_pred": prediction, "baseline_y_pred": baseline,
                        "residual_target": truth - prediction,
                    })
                    writer = _write_prediction(writer, prediction_path, prediction_frame)
                cut_rows.append({
                    "cut_id": str(cut.cut_id), "well_id": well_id,
                    "requested_fraction": float(cut.requested_fraction),
                    "evaluation_role": str(cut.evaluation_role), "cut_index": cut_index,
                    "original_public_cut_index": original_cut, "stage16_fold": int(cut.fold),
                    "public_fold": public_fold, "suffix_rows": len(truth),
                    "replay_eligible": bool(eligible), "baseline_sse": baseline_sse,
                    "public_sse": public_sse, "hybrid_sse": hybrid_sse,
                })
            if position % 100 == 0:
                print(f"replayed {position}/{len(selected_wells)} wells", flush=True)
    finally:
        if writer is not None:
            writer.close()

    cuts = pd.DataFrame.from_records(cut_rows)
    cuts["baseline_rmse"] = np.sqrt(cuts["baseline_sse"] / cuts["suffix_rows"])
    cuts["public_rmse"] = np.sqrt(cuts["public_sse"] / cuts["suffix_rows"])
    cuts["hybrid_rmse"] = np.sqrt(cuts["hybrid_sse"] / cuts["suffix_rows"])
    cuts.to_parquet(output / "cut_report.parquet", index=False)
    role_report = {role: _summarize_cuts(cuts, role) for role in ["primary", "diagnostic"]}
    primary_folds = _fold_report(cuts, "primary")
    gates_config = dict(config.get("gates", {}))
    improved_public_folds = sum(row["eligible_public_delta"] < 0 for row in primary_folds)
    improved_hybrid_folds = sum(row["hybrid_delta"] < 0 for row in primary_folds)
    gates = {
        "manifest_frozen": True,
        "public_oof_well_isolated": True,
        "target_alignment": maximum_target_difference <= float(config["provenance"].get("target_tolerance", 1e-5)),
        "minimum_primary_coverage": role_report["primary"]["row_coverage"] >= float(gates_config.get("minimum_primary_row_coverage", 0.35)),
        "eligible_public_gain": role_report["primary"]["eligible_public_delta"] <= -float(gates_config.get("minimum_eligible_gain", 0.05)),
        "eligible_fold_consistency": improved_public_folds >= int(gates_config.get("minimum_improved_folds", 4)),
        "hybrid_primary_gain": role_report["primary"]["hybrid_delta"] < 0,
    }
    summary = {
        "stage17a_complete": True,
        "promoted_to_selector_replay": bool(all(gates.values())),
        "source_branch": "ravaghi_public_groupkfold_oof",
        "stage16b_manifest_sha256": expected_manifest,
        "public_oof_file_sha256": _sha256(public_path),
        "n_wells": len(selected_wells), "n_cuts": len(cuts),
        "maximum_target_alignment_difference": maximum_target_difference,
        "role_report": role_report, "primary_fold_report": primary_folds,
        "improved_eligible_folds": f"{improved_public_folds}/{len(primary_folds)}",
        "improved_hybrid_folds": f"{improved_hybrid_folds}/{len(primary_folds)}",
        "gates": gates,
        "unvalidated_v599_components": [
            "SP45 selector PF", "projection", "learned trajectory branch",
            "visible-prefix calibration", "model-package correction", "PF seed-branch hedge",
        ],
        "next_step": "Replay the deterministic SP45 selector on uncovered short-prefix cuts.",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "provenance.json", {
        "stage16b_run": str(stage16_run), "public_oof_run": str(public_run),
        "public_oof_rows": len(public), "public_oof_folds": int(public["fold"].nunique()),
        "public_oof_fold_constant_per_well": True,
        "replay_rule": "original_public_cut_index <= pseudo_cut_index",
        "same_well_target_leakage_guard": True,
    })
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16_run), "public_oof_run": str(public_run),
        "data_dir": str(args.data_dir.resolve()), "run_id": args.run_id,
        "limit_wells": args.limit_wells,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
