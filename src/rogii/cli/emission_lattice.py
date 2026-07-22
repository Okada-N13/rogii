from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.config import load_config
from rogii.data.loading import discover_horizontal_wells, load_horizontal_well, load_typewell, well_id_from_path
from rogii.evaluation.delta_u_gate import absolute_tail_metrics, nested_select_predictions, prediction_report
from rogii.evaluation.metrics import evaluate_predictions, paired_well_bootstrap
from rogii.models.emission_features import EmissionSequence, build_emission_sequence, feature_invariance
from rogii.models.emission_lattice import decode_lattice
from rogii.models.emission_tcn import CandidateEmissionTCN, predict_emissions, train_emission_fold
from rogii.models.raw_ncc import offset_grid


FAMILIES = ("spatial_fold", "typewell_fold", "fold")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 12C spatial emission and K-best lattice audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage11-run", type=Path, required=True)
    parser.add_argument("--stage11c-run", type=Path, required=True)
    parser.add_argument("--stage12a-run", type=Path, required=True)
    parser.add_argument("--stage12b-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--limit-wells", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _load_wells(data_dir: Path, selected: set[str]):
    horizontal, typewell = {}, {}
    for path in discover_horizontal_wells(data_dir, "train"):
        well_id = well_id_from_path(path)
        if well_id in selected:
            horizontal[well_id] = load_horizontal_well(path)
            typewell[well_id] = load_typewell(path)
    missing = selected - set(horizontal)
    if missing:
        raise FileNotFoundError(f"Missing Stage 12C wells: {sorted(missing)[:3]}")
    return horizontal, typewell


def _fold_value(item: EmissionSequence, family: str) -> int:
    return int(getattr(item, family))


def _load_model(path: Path, offsets: np.ndarray, device: torch.device) -> CandidateEmissionTCN:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not np.array_equal(np.asarray(checkpoint["offsets"]), offsets):
        raise RuntimeError(f"Checkpoint offset mismatch: {path}")
    model = CandidateEmissionTCN(
        int(checkpoint["n_costs"]), int(checkpoint["n_row_features"]), offsets,
        dict(checkpoint["model_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _save_model(path: Path, model: CandidateEmissionTCN, item: EmissionSequence, offsets: np.ndarray, config: dict[str, Any]) -> None:
    torch.save(
        {
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "offsets": offsets,
            "model_config": config,
            "n_costs": int(item.costs.shape[1]),
            "n_row_features": int(item.row_features.shape[1]),
        },
        path,
    )


def _family_logits(
    family: str,
    sequences: list[EmissionSequence],
    offsets: np.ndarray,
    model_config: dict[str, Any],
    seed: int,
    device: torch.device,
    output: Path,
    stage12b: Path,
    resume: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    logits: dict[str, np.ndarray] = {}
    histories: dict[str, Any] = {}
    folds = sorted({_fold_value(item, family) for item in sequences})
    for fold in folds:
        valid = [item for item in sequences if _fold_value(item, family) == fold]
        if family == "fold":
            checkpoint_path = stage12b / f"fold_{fold}.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Missing Stage 12B checkpoint: {checkpoint_path}")
            model = _load_model(checkpoint_path, offsets, device)
            history_path = stage12b / f"fold_{fold}_history.json"
            history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
        else:
            train = [item for item in sequences if _fold_value(item, family) != fold]
            checkpoint_path = output / f"{family}_{fold}.pt"
            history_path = output / f"{family}_{fold}_history.json"
            if resume and checkpoint_path.is_file():
                model = _load_model(checkpoint_path, offsets, device)
                history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
                print(f"Reusing {family} checkpoint {fold}", flush=True)
            else:
                model, history = train_emission_fold(train, valid, offsets, model_config, seed + 100 * (1 + FAMILIES.index(family)) + fold, device)
                _save_model(checkpoint_path, model, train[0], offsets, model_config)
                write_json(history_path, history)
        histories[str(fold)] = history
        predictions = predict_emissions(model, valid, device)
        for item, values in zip(valid, predictions, strict=True):
            logits[item.cut_id] = values.astype(np.float32)
        print({"family": family, "fold": fold, "cuts": len(valid), "epochs": len(history)}, flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return logits, histories


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def _family_predictions(
    family: str,
    sequences: list[EmissionSequence],
    logits: dict[str, np.ndarray],
    offsets: np.ndarray,
    profiles: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    base_parts: list[pd.DataFrame] = []
    path_values = {str(profile["name"]): [] for profile in profiles}
    for index, item in enumerate(sequences, 1):
        values = logits[item.cut_id]
        probabilities = _softmax(values)
        expected = probabilities @ offsets
        base_parts.append(
            pd.DataFrame(
                {
                    "id": item.cut_id + "_" + item.row_index.astype(str),
                    "well_id": item.well_id,
                    "cut_id": item.cut_id,
                    "cut_fraction": item.cut_fraction,
                    "row_index": item.row_index,
                    "MD": item.md,
                    "y_true": item.y_true,
                    "surface_y_pred": item.surface_y_pred,
                    "expected_offset": expected,
                    "y_pred": item.surface_y_pred + expected,
                    "fold": _fold_value(item, family),
                }
            )
        )
        for profile in profiles:
            decoded = decode_lattice(values, offsets, profile)
            weight = float(profile.get("blend_weight", 1.0))
            correction = (1.0 - weight) * expected + weight * np.asarray(decoded["posterior_mean"])
            path_values[str(profile["name"])].append(item.surface_y_pred + correction)
        if index % 200 == 0:
            print(f"{family}: decoded {index}/{len(sequences)} cuts", flush=True)
    base = pd.concat(base_parts, ignore_index=True)
    return base, {name: np.concatenate(parts) for name, parts in path_values.items()}


def _family_report(base: pd.DataFrame, predictions: dict[str, np.ndarray], selection: dict[str, Any]):
    nested, selections = nested_select_predictions(base, predictions, selection)
    base_metrics, _ = evaluate_predictions(base)
    nested_metrics, _ = evaluate_predictions(nested)
    profile_reports = {name: prediction_report(base, values) for name, values in predictions.items()}
    bootstrap = paired_well_bootstrap(nested, base, n_resamples=int(selection.get("bootstrap_resamples", 1000)), seed=42)
    return nested, selections, {
        "base_metrics": base_metrics,
        "nested_metrics": nested_metrics,
        "nested_rmse_delta": float(nested_metrics["pooled_rmse"] - base_metrics["pooled_rmse"]),
        "nested_tail": absolute_tail_metrics(nested),
        "base_tail": absolute_tail_metrics(base),
        "bootstrap": bootstrap,
        "selections": selections,
        "profile_reports": profile_reports,
    }


def _choose_inference_profile(family_reports: dict[str, Any], profiles: list[dict[str, Any]], tolerance: float) -> str | None:
    eligible: list[tuple[float, str]] = []
    for profile in profiles:
        name = str(profile["name"])
        reports = [family_reports[family]["profile_reports"][name] for family in FAMILIES]
        if all(report["pooled_rmse_delta"] < 0.0 and max(report["fold_deltas"].values()) <= tolerance for report in reports):
            eligible.append((sum(float(report["pooled_rmse_delta"]) for report in reports), name))
    return min(eligible)[1] if eligible else None


def _standard_kbest(
    sequences: list[EmissionSequence],
    logits: dict[str, np.ndarray],
    offsets: np.ndarray,
    profiles: dict[str, dict[str, Any]],
    selections: list[dict[str, Any]],
    base: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = {int(row["fold"]): row["selected_spec"] for row in selections}
    parts: dict[str, list[np.ndarray]] = {
        "posterior_mean": [], "posterior_map": [], "viterbi": [], "kbest": []
    }
    effective = []
    for item in sequences:
        name = selected.get(item.fold)
        probabilities = _softmax(logits[item.cut_id])
        expected = probabilities @ offsets
        if name is None:
            corrections = {key: expected for key in parts}
        else:
            profile = profiles[str(name)]
            decoded = decode_lattice(logits[item.cut_id], offsets, profile, k_best=k)
            weight = float(profile.get("blend_weight", 1.0))
            corrections = {
                "posterior_mean": (1.0 - weight) * expected + weight * np.asarray(decoded["posterior_mean"]),
                "posterior_map": (1.0 - weight) * expected + weight * np.asarray(decoded["posterior_map"]),
                "viterbi": (1.0 - weight) * expected + weight * np.asarray(decoded["viterbi"]),
                "kbest": (1.0 - weight) * expected + weight * np.asarray(decoded["kbest_mean"]),
            }
            effective.append(float(decoded["kbest_effective_paths"]))
        for key, correction in corrections.items():
            parts[key].append(item.surface_y_pred + correction)
    candidate = base.copy()
    comparison: dict[str, Any] = {}
    base_metrics, _ = evaluate_predictions(base)
    for key, values in parts.items():
        candidate[f"{key}_y_pred"] = np.concatenate(values)
        evaluation = base.copy()
        evaluation["y_pred"] = candidate[f"{key}_y_pred"].to_numpy()
        metrics, _ = evaluate_predictions(evaluation)
        comparison[key] = {
            "pooled_rmse": float(metrics["pooled_rmse"]),
            "rmse_delta": float(metrics["pooled_rmse"] - base_metrics["pooled_rmse"]),
            "well_rmse_p90": float(metrics["well_rmse_p90"]),
            "worst_10pct_sse_share": float(metrics["worst_10pct_sse_share"]),
        }
    candidate["y_pred"] = candidate["kbest_y_pred"]
    metrics, _ = evaluate_predictions(candidate)
    return candidate, {
        "metrics": metrics,
        "rmse_delta": float(metrics["pooled_rmse"] - base_metrics["pooled_rmse"]),
        "tail": absolute_tail_metrics(candidate),
        "decoder_comparison": comparison,
        "mean_effective_paths": float(np.mean(effective)) if effective else 0.0,
        "k": int(k),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    ncc = dict(config.get("ncc", {}))
    model_config = dict(config.get("model", {}))
    selection = dict(config.get("selection", {}))
    validation = dict(config.get("validation", {}))
    profiles = [dict(value) for value in config.get("profiles", [])]
    if not profiles or len({profile["name"] for profile in profiles}) != len(profiles):
        raise ValueError("Stage 12C requires uniquely named path profiles")
    stage11, stage11c = args.stage11_run.resolve(), args.stage11c_run.resolve()
    stage12a, stage12b = args.stage12a_run.resolve(), args.stage12b_run.resolve()
    summary11c = load_config(stage11c / "gate_summary.json")
    summary12a = load_config(stage12a / "benchmark_summary.json")
    summary12b = load_config(stage12b / "gate_summary.json")
    if not summary11c.get("promoted_to_stage12") or not summary12a.get("promoted_to_learned_emission") or not summary12b.get("promoted_to_spatial_emission_audit"):
        raise RuntimeError("Stages 11C, 12A, and 12B must authorize Stage 12C")
    surface = dict(summary11c["selected_inference_parameters"])
    output = args.artifact_dir.resolve() / args.run_id
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    offsets = offset_grid(ncc)
    family_reports, histories = {}, {}
    standard_cache = None
    hidden_checks = []
    selected_wells: list[str] | None = None

    for family in FAMILIES:
        coefficients = pd.read_parquet(stage11 / f"{family}_coefficient_oof.parquet")
        wells = sorted(coefficients["well_id"].astype(str).unique())
        if args.limit_wells:
            wells = wells[: args.limit_wells]
            coefficients = coefficients[coefficients["well_id"].astype(str).isin(set(wells))].copy()
        if selected_wells is None:
            selected_wells = wells
        horizontal, typewell = _load_wells(args.data_dir.resolve(), set(wells))
        if family == "fold":
            for record in coefficients.head(min(6, len(coefficients))).itertuples(index=False):
                hidden_checks.append({"cut_id": str(record.cut_id), "passed": feature_invariance(record, horizontal[str(record.well_id)], typewell[str(record.well_id)], ncc, weight=float(surface["weight"]), correction_cap_ft=float(surface["cap"]))})
        sequences = []
        for index, record in enumerate(coefficients.itertuples(index=False), 1):
            sequences.append(build_emission_sequence(record, horizontal[str(record.well_id)], typewell[str(record.well_id)], ncc, weight=float(surface["weight"]), correction_cap_ft=float(surface["cap"])))
            if index % 100 == 0:
                print(f"{family}: built {index}/{len(coefficients)} sequences", flush=True)
        logits, history = _family_logits(family, sequences, offsets, model_config, int(config.get("seed", 42)), device, output, stage12b, args.resume)
        del horizontal, typewell, coefficients
        histories[family] = history
        base, path_predictions = _family_predictions(family, sequences, logits, offsets, profiles)
        nested, selections, report = _family_report(base, path_predictions, selection)
        family_reports[family] = report
        artifact = base.copy()
        artifact["nested_y_pred"] = nested["y_pred"].to_numpy()
        for name, values in path_predictions.items():
            artifact[f"{name}_y_pred"] = values
        artifact.to_parquet(output / f"{family}_path_oof.parquet", index=False)
        print({"family": family, "base_rmse": report["base_metrics"]["pooled_rmse"], "nested_rmse": report["nested_metrics"]["pooled_rmse"], "delta": report["nested_rmse_delta"]}, flush=True)
        if family == "fold":
            standard_cache = (sequences, logits, base, selections)
        else:
            del sequences, logits, base, path_predictions, nested

    hidden_invariance = bool(hidden_checks) and all(row["passed"] for row in hidden_checks)
    inference_name = _choose_inference_profile(family_reports, profiles, float(validation.get("inference_fold_tolerance", 0.05)))
    profile_map = {str(profile["name"]): profile for profile in profiles}
    assert standard_cache is not None
    standard_sequences, standard_logits, standard_base, standard_selections = standard_cache
    kbest_frame, kbest_report = _standard_kbest(standard_sequences, standard_logits, offsets, profile_map, standard_selections, standard_base, int(validation.get("k_best", 16)))
    kbest_frame.to_parquet(output / "fold_kbest_nested_oof.parquet", index=False)
    standard = family_reports["fold"]
    spatial = family_reports["spatial_fold"]
    typewell = family_reports["typewell_fold"]
    gates = {
        "hidden_target_invariance": hidden_invariance,
        "standard_nested_gain": standard["nested_rmse_delta"] <= -float(validation.get("minimum_standard_gain", 0.05)),
        "spatial_nested_gain": spatial["nested_rmse_delta"] <= -float(validation.get("minimum_spatial_gain", 0.02)),
        "typewell_nested_gain": typewell["nested_rmse_delta"] <= -float(validation.get("minimum_typewell_gain", 0.02)),
        "standard_bootstrap": standard["bootstrap"]["ci_97_5"] < 0.0,
        "spatial_bootstrap": spatial["bootstrap"]["ci_97_5"] < 0.0,
        "typewell_bootstrap": typewell["bootstrap"]["ci_97_5"] < 0.0,
        "standard_p90_nonworse": standard["nested_metrics"]["well_rmse_p90"] <= standard["base_metrics"]["well_rmse_p90"] * (1.0 + float(validation.get("tail_tolerance", 0.01))),
        "standard_worst10_nonworse": standard["nested_metrics"]["worst_10pct_sse_share"] <= standard["base_metrics"]["worst_10pct_sse_share"] * (1.0 + float(validation.get("tail_tolerance", 0.01))),
        "kbest_nested_nonworse": kbest_report["rmse_delta"] <= float(validation.get("kbest_tolerance", 0.02)),
        "robust_inference_profile": inference_name is not None,
    }
    promoted = all(gates.values())
    summary = {
        "promoted_to_all_train_alignment": promoted,
        "experiment": "stage12c_spatial_kbest_lattice",
        "surface_spec": surface,
        "device": str(device),
        "n_wells": len(selected_wells or []),
        "offset_states": len(offsets),
        "family_reports": family_reports,
        "standard_kbest": kbest_report,
        "inference_profile": profile_map.get(inference_name) if inference_name else None,
        "hidden_target_invariance": hidden_checks,
        "gates": gates,
        "next_step": "Train all-well emission ensemble and export the independent inference package." if promoted else "Revise path profiles or emission spatial generalization before inference export.",
    }
    write_json(output / "training_histories.json", histories)
    write_json(output / "gate_summary.json", summary)
    write_json(output / "environment.json", environment_report())
    config["resolved"] = {"stage11_run": str(stage11), "stage11c_run": str(stage11c), "stage12a_run": str(stage12a), "stage12b_run": str(stage12b), "data_dir": str(args.data_dir.resolve()), "run_id": args.run_id, "limit_wells": args.limit_wells, "resume": args.resume}
    write_yaml(output / "config.yaml", config)
    print({"promoted_to_all_train_alignment": promoted, "standard_delta": standard["nested_rmse_delta"], "spatial_delta": spatial["nested_rmse_delta"], "typewell_delta": typewell["nested_rmse_delta"], "standard_kbest_delta": kbest_report["rmse_delta"], "inference_profile": summary["inference_profile"], "gates": gates, "next_step": summary["next_step"]}, flush=True)
    print(f"run artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
