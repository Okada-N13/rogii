from __future__ import annotations

import argparse
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.branch_retrieval import _bootstrap, _metrics
from rogii.cli.expanded_residual_field import (
    expanded_residual_features,
    smooth_residual_target,
)
from rogii.cli.prefix_residual_field import _correction_prediction
from rogii.cli.prefix_router import FOLD_FAMILIES, _family_report, build_candidates
from rogii.cli.trajectory_residual import _typewell_folds
from rogii.config import load_config
from rogii.models.residual_tcn import ResidualTCN


class ChunkDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, sequences: list[dict[str, Any]], chunk_rows: int, overlap_rows: int):
        self.sequences = sequences
        self.chunk_rows = int(chunk_rows)
        step = max(1, self.chunk_rows - int(overlap_rows))
        self.index: list[tuple[int, int]] = []
        for sequence_index, sequence in enumerate(sequences):
            length = len(sequence["y"])
            starts = list(range(0, max(1, length - self.chunk_rows + 1), step))
            last = max(0, length - self.chunk_rows)
            if not starts or starts[-1] != last:
                starts.append(last)
            self.index.extend((sequence_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_index, start = self.index[index]
        item = self.sequences[sequence_index]
        stop = min(start + self.chunk_rows, len(item["y"]))
        length = stop - start
        features = np.zeros((self.chunk_rows, item["x"].shape[1]), np.float32)
        target = np.zeros(self.chunk_rows, np.float32)
        mask = np.zeros(self.chunk_rows, np.float32)
        features[:length] = item["x"][start:stop]
        target[:length] = item["y"][start:stop]
        mask[:length] = 1.0
        return torch.from_numpy(features), torch.from_numpy(target), torch.from_numpy(mask)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 30A smooth residual TCN")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--public-oof-run", type=Path, required=True)
    parser.add_argument("--split-manifest-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit-training-cuts", type=int)
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def _predict(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(features[None].astype(np.float32)).to(device)
        return model(tensor).squeeze(0).float().cpu().numpy()


def _make_model(feature_count: int, config: dict[str, Any]) -> ResidualTCN:
    return ResidualTCN(
        feature_count,
        channels=int(config.get("channels", 48)),
        blocks=int(config.get("blocks", 5)),
        kernel_size=int(config.get("kernel_size", 5)),
        dropout=float(config.get("dropout", 0.1)),
    )


def _train_fold(
    fold: int,
    training: list[dict[str, Any]],
    model_config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[ResidualTCN, list[dict[str, float]]]:
    train = [item for item in training if item["fold"] != fold]
    valid = [item for item in training if item["fold"] == fold]
    dataset = ChunkDataset(
        train, int(model_config.get("chunk_rows", 256)),
        int(model_config.get("chunk_overlap_rows", 64)),
    )
    generator = torch.Generator().manual_seed(seed + fold)
    loader = DataLoader(
        dataset, batch_size=int(model_config.get("batch_size", 16)),
        shuffle=True, generator=generator, num_workers=0,
    )
    model = _make_model(training[0]["x"].shape[1], model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config.get("learning_rate", 8e-4)),
        weight_decay=float(model_config.get("weight_decay", 1e-4)),
    )
    beta = float(model_config.get("huber_beta", 0.5))
    best_state = None
    best_rmse = float("inf")
    patience = 0
    history = []
    for epoch in range(int(model_config.get("epochs", 10))):
        model.train()
        train_loss = 0.0
        train_rows = 0.0
        for features, target, mask in loader:
            features, target, mask = features.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss_values = nn.functional.smooth_l1_loss(
                prediction, target, reduction="none", beta=beta
            )
            loss = (loss_values * mask).sum() / mask.sum().clamp_min(1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(model_config.get("gradient_clip", 1.0)))
            optimizer.step()
            train_loss += float((loss_values * mask).sum().detach().cpu())
            train_rows += float(mask.sum().detach().cpu())
        valid_sse, valid_rows = 0.0, 0
        for item in valid:
            prediction = _predict(model, item["x"], device)
            valid_sse += float(np.square(prediction - item["y"]).sum())
            valid_rows += len(item["y"])
        valid_rmse = float(np.sqrt(valid_sse / max(valid_rows, 1)))
        history.append(
            {
                "epoch": epoch + 1,
                "train_huber": train_loss / max(train_rows, 1.0),
                "valid_scaled_residual_rmse": valid_rmse,
            }
        )
        print({"fold": fold, **history[-1]}, flush=True)
        if valid_rmse < best_rmse - 1e-5:
            best_rmse = valid_rmse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(model_config.get("patience", 2)):
                break
    if best_state is None:
        raise RuntimeError(f"Fold {fold} did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(args.device)
    expected_hash = str(config["provenance"]["stage16b_manifest_sha256"])
    stage16, stage17 = args.stage16b_run.resolve(), args.stage17a_run.resolve()
    public_run = args.public_oof_run.resolve()
    manifest_run, validation_run = args.split_manifest_run.resolve(), args.validation_run.resolve()
    manifest = json.loads((manifest_run / "summary.json").read_text(encoding="utf-8"))
    if manifest.get("training_wells") != 500 or manifest.get("confirmation_wells") != 120:
        raise AssertionError("Stage 30A requires the frozen 500/120 multi-cut split")
    if manifest.get("reserved_confirmation_used") is not False:
        raise AssertionError("Stage 30A manifest opened the confirmation reserve")
    summary17 = json.loads((stage17 / "summary.json").read_text(encoding="utf-8"))
    if summary17.get("stage16b_manifest_sha256") != expected_hash:
        raise AssertionError("Stage 17A provenance mismatch")
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_parquet(stage17 / "cut_report.parquet")
    training_ids = pd.read_parquet(manifest_run / "training_cut_ids.parquet", columns=["cut_id"])
    validation_ids = pd.read_parquet(validation_run / "confidence_cut_report.parquet", columns=["cut_id"])
    training_cuts = cuts[cuts["cut_id"].isin(training_ids["cut_id"])].copy().sort_values("cut_id")
    validation_cuts = cuts[cuts["cut_id"].isin(validation_ids["cut_id"])].copy().sort_values("cut_id")
    if args.limit_training_cuts is not None:
        training_cuts = training_cuts.head(int(args.limit_training_cuts)).copy()
    training_wells = set(training_cuts["well_id"].astype(str))
    validation_wells = set(validation_cuts["well_id"].astype(str))
    if training_wells & validation_wells:
        raise AssertionError("Training/design-validation well overlap")
    assignments = pd.read_parquet(stage16 / "well_assignments.parquet")
    assignments["well_id"] = assignments["well_id"].astype(str)
    assignments["typewell_fold"] = _typewell_folds(assignments, 5, seed)
    columns = ["well_id", "spatial_fold", "typewell_fold", "branch_group_fold"]
    training_cuts = training_cuts.merge(assignments[columns], on="well_id", how="left", validate="many_to_one")
    validation_cuts = validation_cuts.merge(assignments[columns], on="well_id", how="left", validate="many_to_one")
    for frame in (training_cuts, validation_cuts):
        frame["stage16_fold"] = frame["stage16_fold"].astype(int)
    wells = sorted(training_wells | validation_wells)
    public = pd.read_parquet(
        public_run / "base_oof.parquet", columns=["well_id", "row_index", "y_pred"],
        filters=[("well_id", "in", wells)],
    )
    public["well_id"] = public["well_id"].astype(str)
    public_by_well = {well: frame.sort_values("row_index") for well, frame in public.groupby("well_id")}
    train_dir = args.data_dir.resolve() / "train"

    @lru_cache(maxsize=32)
    def load_well(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train_dir / f"{well_id}__horizontal_well.csv")

    @lru_cache(maxsize=32)
    def load_typewell(well_id: str) -> pd.DataFrame:
        return pd.read_csv(train_dir / f"{well_id}__typewell.csv")

    candidate_config = dict(config.get("candidates", {}))
    feature_config = dict(config.get("features", {}))
    model_config = dict(config.get("model", {}))
    base_name = str(model_config.get("base_candidate", "top_pf_a130"))
    stride = int(model_config.get("training_stride", 8))
    target_scale = float(model_config.get("target_scale_ft", 10.0))
    feature_names = None
    training = []
    for position, cut in enumerate(training_cuts.itertuples(index=False), 1):
        well_id, outer = str(cut.well_id), int(cut.cut_index)
        horizontal, typewell = load_well(well_id), load_typewell(well_id)
        source = public_by_well[well_id]
        original = int(source["row_index"].min())
        candidates = build_candidates(
            horizontal, typewell, outer,
            source["y_pred"].to_numpy(float)[outer - original :], candidate_config,
        )
        features, names = expanded_residual_features(
            horizontal, outer, candidates, base_name, float(cut.requested_fraction),
            plane_window_ft=float(feature_config.get("plane_window_ft", 1200.0)),
            plane_ridge=float(feature_config.get("plane_ridge", 0.05)),
        )
        feature_names = names if feature_names is None else feature_names
        if names != feature_names:
            raise AssertionError("Feature schema changed")
        raw_target = horizontal["TVT"].to_numpy(float)[outer:] - candidates[base_name]
        target = smooth_residual_target(
            raw_target, int(model_config.get("target_smoothing_rows", 101)),
            float(model_config.get("target_cap_ft", 24.0)),
        )
        index = np.arange(0, len(features), stride, dtype=int)
        training.append(
            {
                "fold": int(cut.stage16_fold),
                "x_raw": features[index],
                "y": target[index] / target_scale,
            }
        )
        if position % 25 == 0:
            print(f"residual TCN features {position}/{len(training_cuts)} cuts", flush=True)
    stacked = np.concatenate([item["x_raw"] for item in training])
    feature_mean = stacked.mean(axis=0).astype(np.float32)
    feature_scale = np.maximum(stacked.std(axis=0), 1e-4).astype(np.float32)
    for item in training:
        item["x"] = ((item.pop("x_raw") - feature_mean) / feature_scale).astype(np.float32)
        item["y"] = np.asarray(item["y"], np.float32)
    np.savez(output / "normalizer.npz", mean=feature_mean, scale=feature_scale)

    models = []
    histories = {}
    for fold in sorted(training_cuts["stage16_fold"].unique()):
        model, history = _train_fold(int(fold), training, model_config, device, seed)
        checkpoint = {
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "feature_count": len(feature_names),
            "model_config": model_config,
            "feature_names": feature_names,
        }
        torch.save(checkpoint, output / f"fold_{int(fold)}.pt")
        models.append(model)
        histories[str(int(fold))] = history
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    correction = dict(config.get("correction", {}))
    weights = [float(value) for value in correction.get("diagnostic_weights", [0.1, 0.2, 0.3])]
    primary_weight = float(correction.get("primary_weight", 0.2))
    rows = []
    invariance = []
    for position, cut in enumerate(validation_cuts.itertuples(index=False), 1):
        well_id, outer = str(cut.well_id), int(cut.cut_index)
        horizontal, typewell = load_well(well_id), load_typewell(well_id)
        source = public_by_well[well_id]
        original = int(source["row_index"].min())
        candidates = build_candidates(
            horizontal, typewell, outer,
            source["y_pred"].to_numpy(float)[outer - original :], candidate_config,
        )
        features, names = expanded_residual_features(
            horizontal, outer, candidates, base_name, float(cut.requested_fraction),
            plane_window_ft=float(feature_config.get("plane_window_ft", 1200.0)),
            plane_ridge=float(feature_config.get("plane_ridge", 0.05)),
        )
        if names != feature_names:
            raise AssertionError("Validation feature schema changed")
        normalized = ((features - feature_mean) / feature_scale).astype(np.float32)
        raw_residual = target_scale * np.mean(
            [_predict(model, normalized, device) for model in models], axis=0
        )
        base = candidates[base_name]
        truth = horizontal["TVT"].to_numpy(float)[outer:]
        row: dict[str, Any] = {
            "cut_id": str(cut.cut_id), "well_id": well_id,
            "requested_fraction": float(cut.requested_fraction),
            **{family: int(getattr(cut, family)) for family in FOLD_FAMILIES},
            "suffix_rows": len(truth), "base_sse": float(np.square(base - truth).sum()),
            "raw_residual_mean": float(np.mean(raw_residual)),
            "raw_residual_std": float(np.std(raw_residual)),
        }
        for weight in weights:
            prediction = _correction_prediction(
                base, raw_residual, weight, float(correction.get("cap_ft", 8.0)),
                float(correction.get("ramp_rows", 96.0)),
            )
            row[f"candidate_sse_w{int(round(weight * 1000)):03d}"] = float(np.square(prediction - truth).sum())
        rows.append(row)
        if len(invariance) < 8:
            changed = horizontal.copy()
            changed.loc[changed.index >= outer, "TVT"] += 9999.0
            changed_candidates = build_candidates(
                changed, typewell, outer,
                source["y_pred"].to_numpy(float)[outer - original :], candidate_config,
            )
            changed_features, _ = expanded_residual_features(
                changed, outer, changed_candidates, base_name, float(cut.requested_fraction),
                plane_window_ft=float(feature_config.get("plane_window_ft", 1200.0)),
                plane_ridge=float(feature_config.get("plane_ridge", 0.05)),
            )
            invariance.append(np.array_equal(features, changed_features))
        if position % 10 == 0:
            print(f"residual TCN validation {position}/{len(validation_cuts)} cuts", flush=True)
    report = pd.DataFrame(rows)
    report["candidate_sse"] = report[f"candidate_sse_w{int(round(primary_weight * 1000)):03d}"]
    metrics = _metrics(report)
    bootstrap = _bootstrap(report, int(config.get("validation", {}).get("bootstrap_resamples", 4000)), seed)
    family_reports = {family: _family_report(report, family) for family in FOLD_FAMILIES}
    total_rows = int(report["suffix_rows"].sum())
    base_rmse = float(np.sqrt(report["base_sse"].sum() / total_rows))
    weight_report = []
    for weight in weights:
        candidate = float(np.sqrt(report[f"candidate_sse_w{int(round(weight * 1000)):03d}"].sum() / total_rows))
        weight_report.append({"weight": weight, "base_rmse": base_rmse, "candidate_rmse": candidate, "rmse_delta": candidate - base_rmse})
    validation_config = dict(config.get("validation", {}))
    gates = {
        "hidden_target_invariance": bool(invariance) and all(invariance),
        "training_validation_well_overlap_zero": not bool(training_wells & validation_wells),
        "primary_gain": metrics["rmse_delta"] <= -float(validation_config.get("minimum_gain", 0.10)),
        "bootstrap_upper_below_zero": bootstrap[1] < 0.0,
        "standard_fold_consistency": metrics["improved_folds"] >= 4,
        "fraction_consistency": metrics["improved_fractions"] >= 3,
        "spatial_fold_consistency": family_reports["spatial_fold"]["improved_folds"] >= 4,
        "typewell_fold_consistency": family_reports["typewell_fold"]["improved_folds"] >= 4,
        "branch_fold_consistency": family_reports["branch_group_fold"]["improved_folds"] >= 4,
        "p90_nonworse": metrics["cut_rmse_p90_delta"] <= 0.0,
    }
    promoted = bool(all(gates.values()))
    report.to_parquet(output / "residual_tcn_cut_report.parquet", index=False)
    summary = {
        "stage30a_complete": True,
        "promoted_to_stage30b_reserved_confirmation": promoted,
        "stage16b_manifest_sha256": expected_hash,
        "device": str(device), "training_cuts": len(training_cuts),
        "training_wells": len(training_wells), "training_rows": int(sum(len(item["y"]) for item in training)),
        "validation_cuts": len(validation_cuts), "validation_wells": len(validation_wells),
        "feature_count": len(feature_names), "ensemble_models": len(models),
        "primary_weight": primary_weight, "base_rmse": base_rmse,
        "candidate_rmse": metrics["candidate_rmse"], "rmse_delta": metrics["rmse_delta"],
        "bootstrap_95pct": bootstrap, "cut_rmse_p90_delta": metrics["cut_rmse_p90_delta"],
        "family_reports": family_reports, "weight_report": weight_report,
        "gates": gates, "reserved_confirmation_used": False,
        "next_step": (
            "Run exactly one Stage 30B audit on the frozen 120 confirmation wells."
            if promoted else
            "Reject the smooth residual TCN and keep the confirmation reserve sealed."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "training_history.json", histories)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {
        "stage16b_run": str(stage16), "stage17a_run": str(stage17),
        "public_oof_run": str(public_run), "split_manifest_run": str(manifest_run),
        "validation_run": str(validation_run), "data_dir": str(args.data_dir.resolve()),
        "run_id": args.run_id,
    }
    write_yaml(output / "config.yaml", config)
    print(summary, flush=True)


if __name__ == "__main__":
    main()

