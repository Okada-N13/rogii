from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rogii.data.loading import discover_horizontal_wells, load_horizontal_well, load_typewell, well_id_from_path
from rogii.data.multicut import build_inference_record
from rogii.models.emission_features import build_emission_input
from rogii.models.emission_residual import generic_residual_features, stacked_residual_features
from rogii.models.raw_ncc import offset_grid, surface_prediction
from rogii.models.stage15 import FAMILIES, load_surface_bundle, predict_surface_coefficients


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 15 Internet-OFF competition inference")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_package(root: Path) -> None:
    expected = json.loads((root / "sha256.json").read_text(encoding="utf-8"))
    failures = [name for name, digest in expected.items() if not (root / name).is_file() or _sha256(root / name) != digest]
    if failures:
        raise RuntimeError(f"Stage 15 package integrity failure: {failures[:3]}")


def _load_emission(path: Path, offsets: np.ndarray, device):
    import torch
    from rogii.models.emission_tcn import CandidateEmissionTCN

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not np.array_equal(np.asarray(checkpoint["offsets"]), offsets):
        raise RuntimeError(f"Offset grid mismatch: {path}")
    model = CandidateEmissionTCN(
        int(checkpoint["n_costs"]), int(checkpoint["n_row_features"]), offsets,
        dict(checkpoint["model_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(axis=1, keepdims=True)


def _family_prediction(package: Path, family: str, fold: int, record: dict, horizontal: pd.DataFrame,
                       typewell: pd.DataFrame, manifest: dict, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    bundle = load_surface_bundle(package / "surface" / f"{family}_{fold}.joblib")
    resolved = predict_surface_coefficients(bundle, record)
    item = build_emission_input(
        resolved, horizontal, typewell, dict(manifest["ncc"]),
        weight=float(manifest["surface"]["weight"]),
        correction_cap_ft=float(manifest["surface"]["correction_cap_ft"]),
    )
    checkpoint_name = f"fold_{fold}.pt" if family == "fold" else f"{family}_{fold}.pt"
    model = _load_emission(package / "emission" / checkpoint_name, offset_grid(manifest["ncc"]), device)
    with torch.no_grad():
        costs = torch.from_numpy(item.costs.astype(np.float32))[None].to(device)
        rows = torch.from_numpy(item.row_features)[None].to(device)
        logits = model(costs, rows).squeeze(0).float().cpu().numpy()
    probabilities = _softmax(logits)
    offsets = offset_grid(manifest["ncc"])
    expected_offset = probabilities @ offsets
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
    positions, full_surface = surface_prediction(
        resolved, horizontal,
        weight=float(manifest["surface"]["weight"]),
        correction_cap_ft=float(manifest["surface"]["correction_cap_ft"]),
        slope_cap=float(manifest["ncc"].get("slope_correction_cap", 80.0)),
        curvature_cap=float(manifest["ncc"].get("curvature_cap", 30.0)),
    )
    full_md = horizontal.iloc[positions]["MD"].to_numpy(float)
    sampled_offset = np.interp(full_md, item.md.astype(float), expected_offset)
    full_entropy = np.interp(full_md, item.md.astype(float), entropy)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return full_surface, full_surface + sampled_offset, full_entropy


def _predict_well(package: Path, manifest: dict, assignment: dict | None, horizontal: pd.DataFrame,
                  typewell: pd.DataFrame, device) -> tuple[pd.DataFrame, dict]:
    record = build_inference_record(horizontal, typewell)
    record.update({family: int(assignment[family]) if assignment is not None else -1 for family in FAMILIES})
    predictions, surfaces, entropies = {}, {}, {}
    for family in FAMILIES:
        folds = (
            [int(assignment[family])]
            if assignment is not None
            else [int(value) for value in manifest["available_folds"][family]]
        )
        branches = [
            _family_prediction(package, family, fold, record, horizontal, typewell, manifest, device)
            for fold in folds
        ]
        surfaces[family] = np.mean([value[0] for value in branches], axis=0)
        predictions[family] = np.mean([value[1] for value in branches], axis=0)
        entropies[family] = np.mean([value[2] for value in branches], axis=0)
    cut = int(record["cut_index"])
    suffix = horizontal.iloc[cut:]
    frame = pd.DataFrame({
        "well_id": str(record["well_id"]), "cut_id": str(record["cut_id"]),
        "MD": suffix["MD"].to_numpy(float), "cut_fraction": float(record["cut_fraction"]),
        "surface_y_pred": surfaces["fold"], "y_pred": predictions["fold"],
    })
    generic = generic_residual_features(frame)
    residual_folds = (
        [int(assignment["fold"])]
        if assignment is not None
        else [int(value) for value in manifest["available_folds"]["fold"]]
    )
    raw = np.mean([
        joblib.load(package / "residual" / f"fold_generic_{fold}.joblib").predict(generic.to_numpy(np.float32))
        for fold in residual_folds
    ], axis=0)
    spec = manifest["primary_residual"]
    primary = predictions["fold"] + float(spec["weight"]) * np.clip(raw, -float(spec["cap_ft"]), float(spec["cap_ft"]))

    secondary = None
    stacked_paths = [package / "residual" / f"fold_stacked_{fold}.joblib" for fold in residual_folds]
    if all(path.is_file() for path in stacked_paths):
        stacked = stacked_residual_features(
            frame, predictions["spatial_fold"], predictions["typewell_fold"], entropies["fold"]
        )
        stacked_raw = np.mean([
            joblib.load(path).predict(stacked.to_numpy(np.float32)) for path in stacked_paths
        ], axis=0)
        secondary_spec = manifest["secondary_residual"]
        secondary = predictions["fold"] + float(secondary_spec["weight"]) * np.clip(
            stacked_raw, -float(secondary_spec["cap_ft"]), float(secondary_spec["cap_ft"])
        )
    output = pd.DataFrame({
        "id": str(record["well_id"]) + "_" + suffix["row_index"].astype(str),
        "tvt": primary,
    })
    audit = {
        "well_id": str(record["well_id"]), "cut_index": cut, "rows": len(output),
        "inference_policy": "matched_heldout_fold" if assignment is not None else "all_fold_safe_ensemble",
        "folds": (
            {family: int(assignment[family]) for family in FAMILIES}
            if assignment is not None else manifest["available_folds"]
        ),
        "mean_abs_primary_move": float(np.mean(np.abs(primary - predictions["fold"]))),
        "secondary_available": secondary is not None,
        "primary_secondary_rmse": float(np.sqrt(np.mean(np.square(primary - secondary)))) if secondary is not None else None,
    }
    return output, audit


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    package, data = args.package_dir.resolve(), args.data_dir.resolve()
    verify_package(package)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assignment_rows = manifest.get("well_fold_assignments", manifest["test_well_assignments"])
    assignments = {str(row["well_id"]): row for row in assignment_rows}
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "auto" and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 7:
            device = torch.device("cuda")
        else:
            print(
                f"GPU {torch.cuda.get_device_name(0)} has compute capability {capability}; "
                "falling back to CPU because the Kaggle PyTorch build may not support it.",
                flush=True,
            )
            device = torch.device("cpu")
    else:
        device = torch.device("cpu" if args.device == "auto" else args.device)
    parts, audits = [], []
    for path in discover_horizontal_wells(data, "test"):
        well_id = well_id_from_path(path)
        horizontal, typewell = load_horizontal_well(path), load_typewell(path)
        if "TVT" in horizontal:
            raise AssertionError("Stage 15 test inference must not receive hidden TVT")
        part, audit = _predict_well(package, manifest, assignments.get(well_id), horizontal, typewell, device)
        parts.append(part)
        audits.append(audit)
        print(audit, flush=True)
    prediction = pd.concat(parts, ignore_index=True)
    sample = pd.read_csv(data / "sample_submission.csv")
    aligned = sample[["id"]].merge(prediction, on="id", how="left", validate="one_to_one")
    if aligned["tvt"].isna().any() or not np.isfinite(aligned["tvt"]).all():
        raise RuntimeError("Submission alignment produced missing or non-finite predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(args.output, index=False)
    audit = {
        "rows": len(aligned), "id_order_matches_sample": aligned["id"].equals(sample["id"]),
        "finite_tvt": bool(np.isfinite(aligned["tvt"]).all()), "device": str(device),
        "same_well_target_leakage_guard": True, "wells": audits,
        "supports_unseen_test_wells": True,
        "submission_sha256": _sha256(args.output),
    }
    (args.output.parent / "stage15_inference_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(audit, flush=True)


if __name__ == "__main__":
    main()
