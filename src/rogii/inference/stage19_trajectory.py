"""Standalone, Internet-OFF Stage 19 trajectory residual inference."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


COEFFICIENT_COLUMNS = ("target_residual_level", "target_residual_slope", "target_residual_curve")
HGB_FIELDS = (
    "value", "feature_idx", "num_threshold", "missing_go_to_left",
    "left", "right", "is_leaf", "is_categorical",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {f"{prefix}_{key}": 0.0 for key in ("mean", "std", "q10", "q50", "q90")}
    return {
        f"{prefix}_mean": float(np.mean(values)), f"{prefix}_std": float(np.std(values)),
        f"{prefix}_q10": float(np.quantile(values, .1)),
        f"{prefix}_q50": float(np.quantile(values, .5)),
        f"{prefix}_q90": float(np.quantile(values, .9)),
    }


def _robust_slope(md: np.ndarray, values: np.ndarray, window_ft: float) -> float:
    finite = np.isfinite(md) & np.isfinite(values)
    md, values = md[finite], values[finite]
    if len(md) < 3:
        return 0.0
    use = md >= float(md[-1] - window_ft)
    if int(use.sum()) < 3:
        use = np.ones(len(md), dtype=bool)
    x, y = md[use] - md[use][-1], values[use] - values[use][-1]
    coefficient = float(np.polyfit(x, y, 1)[0])
    residual = y - coefficient * x
    scale = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 1e-6)
    keep = np.abs(residual - np.median(residual)) <= 3.0 * scale
    return float(np.polyfit(x[keep], y[keep], 1)[0]) if int(keep.sum()) >= 3 else coefficient


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3 or np.std(left[finite]) < 1e-8 or np.std(right[finite]) < 1e-8:
        return 0.0
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def _typewell_match(tvt: np.ndarray, gr: np.ndarray, typewell: pd.DataFrame) -> tuple[float, float]:
    tw = typewell[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna().sort_values("TVT")
    tw = tw.groupby("TVT", as_index=False)["GR"].mean()
    if len(tw) < 3:
        return float("nan"), 0.0
    predicted = np.interp(np.asarray(tvt, float), tw.TVT, tw.GR, left=np.nan, right=np.nan)
    gr = np.asarray(gr, float)
    finite = np.isfinite(predicted) & np.isfinite(gr)
    if int(finite.sum()) < 3:
        return float("nan"), 0.0
    return float(np.sqrt(np.mean(np.square(predicted[finite] - gr[finite])))), _correlation(predicted, gr)


def build_feature_record(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    base_prediction: np.ndarray,
    config: dict,
) -> dict[str, float | int | str]:
    """Reproduce the 66 Stage 19 features using only visible test inputs."""
    known = pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
    if not known.any() or known.all():
        raise ValueError("TVT_input must contain a non-empty known prefix and hidden suffix")
    cut = int(np.flatnonzero(~known)[0])
    if not known[:cut].all() or known[cut:].any() or cut < 3 or len(horizontal) - cut < 3:
        raise ValueError("TVT_input must be one valid contiguous prefix")
    prefix, suffix = horizontal.iloc[:cut], horizontal.iloc[cut:]
    anchor, end = prefix.iloc[-1], horizontal.iloc[-1]
    md = prefix.MD.to_numpy(float)
    prefix_tvt = prefix.TVT_input.to_numpy(float)
    prefix_u = prefix_tvt + prefix.Z.to_numpy(float)
    anchor_md = float(anchor.MD)
    dx, dy, dz = float(end.X-anchor.X), float(end.Y-anchor.Y), float(end.Z-anchor.Z)
    horizon_md = float(end.MD-anchor_md)
    full_gr = pd.to_numeric(horizontal.GR, errors="coerce").to_numpy(float)
    prefix_gr, suffix_gr = full_gr[:cut], full_gr[cut:]
    record: dict[str, float | int | str] = {
        "well_id": str(anchor.well_id), "cut_id": f"{anchor.well_id}__cut{cut}",
        "cut_index": cut, "n_rows": len(horizontal), "cut_fraction": cut/len(horizontal),
        "suffix_rows": len(suffix), "anchor_md": anchor_md, "anchor_x": float(anchor.X),
        "anchor_y": float(anchor.Y), "anchor_z": float(anchor.Z),
        "anchor_u": float(prefix_u[-1]), "anchor_tvt": float(prefix_tvt[-1]),
        "prefix_u_slope_per_kft": _robust_slope(md, prefix_u, float(config.get("prefix_window_ft", 800)))*1000,
        "prefix_tvt_slope_per_kft": _robust_slope(md, prefix_tvt, float(config.get("prefix_window_ft", 800)))*1000,
        "horizon_md_ft": horizon_md, "horizon_dx_ft": dx, "horizon_dy_ft": dy,
        "horizon_dz_ft": dz, "horizon_xy_ft": float(np.hypot(dx,dy)),
        "horizon_xyz_ft": float(np.sqrt(dx*dx+dy*dy+dz*dz)),
        "trajectory_dx_per_md": dx/max(horizon_md,1), "trajectory_dy_per_md": dy/max(horizon_md,1),
        "trajectory_dz_per_md": dz/max(horizon_md,1),
        "gr_missing_fraction": float(np.mean(~np.isfinite(full_gr))),
        "gr_prefix_suffix_mean_delta": float(np.nanmean(suffix_gr)-np.nanmean(prefix_gr)),
    }
    record.update(_summary("prefix_gr", prefix_gr))
    record.update(_summary("suffix_gr", suffix_gr))
    record.update(_summary("full_gr", full_gr))
    tw_tvt = pd.to_numeric(typewell.TVT, errors="coerce").to_numpy(float)
    record.update(_summary("typewell_gr", pd.to_numeric(typewell.GR, errors="coerce").to_numpy(float)))
    finite_tw = tw_tvt[np.isfinite(tw_tvt)]
    record.update({
        "typewell_tvt_min": float(np.min(finite_tw)) if len(finite_tw) else 0.,
        "typewell_tvt_max": float(np.max(finite_tw)) if len(finite_tw) else 0.,
        "typewell_tvt_span": float(np.ptp(finite_tw)) if len(finite_tw) else 0.,
        "typewell_rows": len(typewell),
    })
    base = np.asarray(base_prediction, float)
    if len(base) != len(suffix) or not np.isfinite(base).all():
        raise ValueError("submission prediction does not match hidden suffix")
    base_u = base + suffix.Z.to_numpy(float)
    x = np.linspace(0., 1., len(base))
    polynomial = np.polyfit(x, base_u-base_u[0], 2) if len(base) >= 3 else (0.,0.,0.)
    prefix_rmse, prefix_corr = _typewell_match(prefix_tvt, prefix_gr, typewell)
    shifts = [float(v) for v in config.get("typewell_shift_grid_ft", [-40,-30,-20,-10,0,10,20,30,40])]
    matches = [(shift, *_typewell_match(base+shift, suffix_gr, typewell)) for shift in shifts]
    finite_matches = [item for item in matches if np.isfinite(item[1])]
    best_shift, best_rmse, best_corr = min(finite_matches, key=lambda item:(item[1],abs(item[0]))) if finite_matches else (0.,float("nan"),0.)
    zero_rmse = next((item[1] for item in matches if item[0] == 0), best_rmse)
    record.update({
        "base_tvt_start": float(base[0]), "base_tvt_end": float(base[-1]),
        "base_tvt_change": float(base[-1]-base[0]), "base_tvt_std": float(np.std(base)),
        "base_u_start": float(base_u[0]), "base_u_end": float(base_u[-1]),
        "base_u_change": float(base_u[-1]-base_u[0]), "base_u_std": float(np.std(base_u)),
        "base_boundary_jump": float(base[0]-prefix_tvt[-1]),
        "base_u_slope_per_kft": float(polynomial[1]*1000/max(horizon_md,1)),
        "base_u_curve": float(polynomial[0]), "prefix_typewell_gr_rmse": prefix_rmse,
        "prefix_typewell_gr_correlation": prefix_corr, "base_typewell_gr_best_shift": best_shift,
        "base_typewell_gr_best_rmse": best_rmse, "base_typewell_gr_zero_rmse": zero_rmse,
        "base_typewell_gr_shift_gain": float(zero_rmse-best_rmse) if np.isfinite(zero_rmse) and np.isfinite(best_rmse) else 0.,
        "base_typewell_gr_correlation": best_corr,
        "requested_fraction": cut/len(horizontal),
    })
    return record


def _load_model(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        n_trees = int(archive["n_trees"][0])
        return {
            "baseline": float(archive["baseline"][0]), "n_features": int(archive["n_features"][0]),
            "trees": [{field: archive[f"tree_{i:03d}__{field}"].copy() for field in HGB_FIELDS} for i in range(n_trees)],
        }


def _predict(model: dict, features: np.ndarray) -> float:
    row = np.asarray(features, float)
    value = float(model["baseline"])
    for tree in model["trees"]:
        node = 0
        while not bool(tree["is_leaf"][node]):
            feature, current = int(tree["feature_idx"][node]), row[int(tree["feature_idx"][node])]
            left = bool(tree["missing_go_to_left"][node]) if np.isnan(current) else current <= float(tree["num_threshold"][node])
            node = int(tree["left"][node] if left else tree["right"][node])
        value += float(tree["value"][node])
    return value


def _apply(base: np.ndarray, coefficients: np.ndarray, profile: dict) -> np.ndarray:
    step = np.arange(1, len(base)+1, dtype=float)
    x = step/max(len(base),1)
    ramp = 1-np.exp(-step/max(float(profile["ramp_rows"]),1))
    design = np.column_stack((ramp, ramp*x, ramp*(2*x*x-x)))
    correction = np.clip(float(profile["weight"])*(design@coefficients), -float(profile["cap_ft"]), float(profile["cap_ft"]))
    return np.asarray(base,float)+correction


def apply_trajectory_residual(package_dir: Path, data_dir: Path, submission_path: Path) -> dict:
    started = time.perf_counter()
    package_dir, data_dir, submission_path = Path(package_dir), Path(data_dir), Path(submission_path)
    manifest = json.loads((package_dir/"manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("stage19c_trajectory_inference_package") or int(manifest.get("package_version",0)) < 2:
        raise AssertionError("Stage 19C portable package v2 is required")
    feature_columns = list(manifest["feature_columns"])
    models = {target: [] for target in COEFFICIENT_COLUMNS}
    for item in manifest["models"]:
        path = package_dir/item["file"]
        if _sha256(path) != item["sha256"]:
            raise AssertionError(f"Model hash mismatch: {path.name}")
        models[item["target"]].append(_load_model(path))
    submission = pd.read_csv(submission_path)
    if list(submission.columns) != ["id","tvt"]:
        raise AssertionError("submission.csv must have exactly id,tvt columns")
    submission["well_id"] = submission.id.astype(str).str.rsplit("_",n=1).str[0]
    submission["row_index"] = submission.id.astype(str).str.rsplit("_",n=1).str[1].astype(int)
    reports = []
    for well_id, positions in submission.groupby("well_id",sort=False).groups.items():
        horizontal = pd.read_csv(data_dir/"test"/f"{well_id}__horizontal_well.csv")
        typewell = pd.read_csv(data_dir/"test"/f"{well_id}__typewell.csv")
        horizontal["well_id"], horizontal["row_index"] = str(well_id), np.arange(len(horizontal))
        ordered = submission.loc[positions].sort_values("row_index")
        base = ordered.tvt.to_numpy(float)
        feature = build_feature_record(horizontal,typewell,base,manifest["features"])
        row = np.asarray([feature.get(column,np.nan) for column in feature_columns],float)
        coefficients = np.asarray([np.mean([_predict(model,row) for model in models[target]]) for target in COEFFICIENT_COLUMNS])
        candidate = _apply(base,coefficients,manifest["profile"])
        submission.loc[ordered.index,"tvt"] = candidate
        move = candidate-base
        reports.append({"well_id":str(well_id),"status":"applied","rows":len(candidate),
                        "mean_abs_move":float(np.mean(np.abs(move))),"max_abs_move":float(np.max(np.abs(move))),
                        "coefficients":coefficients.tolist()})
    final = submission[["id","tvt"]]
    if len(final) == 0 or not np.isfinite(final.tvt).all():
        raise AssertionError("Stage 19C produced invalid predictions")
    final.to_csv(submission_path,index=False)
    audit = {"stage19_trajectory_applied":True,"rows":len(final),"wells":len(reports),
             "hidden_target_columns_used":False,"well_report":reports,
             "submission_sha256":_sha256(submission_path),"elapsed_seconds":float(time.perf_counter()-started)}
    (submission_path.parent/"stage19_trajectory_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    return audit
