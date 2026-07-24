from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii.artifacts import environment_report, write_json, write_yaml
from rogii.cli.branch_retrieval import _bootstrap, _metrics
from rogii.cli.selector_replay import likelihood_selector
from rogii.cli.selector_resolution import stable_stratified_sample
from rogii.cli.trajectory_base_alignment import top_pf_proxy
from rogii.cli.trajectory_residual import _typewell_folds
from rogii.config import load_config


FOLD_FAMILIES = ("stage16_fold", "spatial_fold", "typewell_fold", "branch_group_fold")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 21A visible-prefix candidate router")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage16b-run", type=Path, required=True)
    parser.add_argument("--stage17a-run", type=Path, required=True)
    parser.add_argument("--public-oof-run", type=Path, required=True)
    parser.add_argument("--exclude-run", type=Path, action="append", default=[])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit-cuts", type=int)
    return parser


def _robust_u_polynomial(horizontal: pd.DataFrame, cut_index: int, degree: int) -> np.ndarray:
    prefix, suffix = horizontal.iloc[:cut_index], horizontal.iloc[cut_index:]
    md = prefix["MD"].to_numpy(float)
    u = prefix["TVT"].to_numpy(float) + prefix["Z"].to_numpy(float)
    origin, scale = float(md[-1]), max(float(md[-1] - md[0]), 1.0)
    x = (md-origin)/scale
    coefficients = np.polyfit(x, u, min(int(degree), len(prefix)-1))
    for _ in range(4):
        residual = u-np.polyval(coefficients,x)
        robust_scale = 1.4826*np.median(np.abs(residual-np.median(residual)))+1e-6
        weights = 1.0/(1.0+np.square(residual/(2.5*robust_scale)))
        coefficients = np.polyfit(x,u,min(int(degree),len(prefix)-1),w=weights)
    future_x = (suffix["MD"].to_numpy(float)-origin)/scale
    return np.polyval(coefficients,future_x)-suffix["Z"].to_numpy(float)


def build_candidates(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cut_index: int,
    public_prediction: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Build candidates without reading TVT at or after ``cut_index``."""
    public_prediction = np.asarray(public_prediction,float)
    if len(public_prediction) != len(horizontal)-int(cut_index):
        raise ValueError("public prediction length mismatch")
    candidates: dict[str,np.ndarray] = {"public_oof":public_prediction}
    selector_base = dict(config.get("selector",{}))
    blend = dict(config.get("top_pf_proxy",{}))
    masked = horizontal[["MD","Z","GR","TVT"]].copy()
    masked.loc[masked.index >= int(cut_index),"TVT"] = np.nan
    for multiplier in [float(value) for value in config.get("gr_sigma_multipliers",[1.0,1.3,1.6])]:
        tag=f"a{int(round(multiplier*100)):03d}"
        selector,_ = likelihood_selector(
            masked,typewell,int(cut_index),
            {**selector_base,"gr_sigma_multiplier":multiplier},
        )
        candidates[f"selector_{tag}"] = selector
        candidates[f"top_pf_{tag}"] = top_pf_proxy(
            horizontal,int(cut_index),public_prediction,selector,blend
        )
    for degree in [int(value) for value in config.get("polynomial_degrees",[1,2,3])]:
        candidates[f"poly_u_deg{degree}"] = _robust_u_polynomial(horizontal,int(cut_index),degree)
    expected = len(horizontal)-int(cut_index)
    if any(len(value)!=expected or not np.isfinite(value).all() for value in candidates.values()):
        raise RuntimeError("Candidate library produced invalid predictions")
    return candidates


def internal_cut_indices(original_cut: int, outer_cut: int, config: dict[str,Any]) -> list[int]:
    minimum_holdout = int(config.get("minimum_holdout_rows",32))
    gap = int(outer_cut)-int(original_cut)
    if gap < int(config.get("minimum_calibration_gap_rows",96)):
        return []
    # Clamp first, then de-duplicate. At the minimum allowed gap the later
    # fraction can fall inside the required outer holdout before clamping;
    # filtering it first incorrectly leaves only one calibration cut.
    maximum_inner = int(outer_cut)-minimum_holdout
    output = {
        max(
            int(original_cut),
            min(
                int(round(int(original_cut)+float(fraction)*gap)),
                maximum_inner,
            ),
        )
        for fraction in config.get("inner_gap_fractions",[.45,.72])
    }
    return sorted(value for value in output if int(original_cut) <= value < int(outer_cut))


def _guarded_route(base: np.ndarray, selected: np.ndarray, config: dict[str,Any]) -> np.ndarray:
    base,selected=np.asarray(base,float),np.asarray(selected,float)
    step=np.arange(1,len(base)+1,dtype=float)
    ramp=1.0-np.exp(-step/max(float(config.get("ramp_rows",96)),1.0))
    move=np.clip(
        float(config.get("blend_weight",.25))*ramp*(selected-base),
        -float(config.get("cap_ft",12)),float(config.get("cap_ft",12)),
    )
    return base+move


def _family_report(frame: pd.DataFrame, family: str) -> dict[str,Any]:
    rows=[]
    for fold,group in frame.groupby(family,sort=True):
        n=int(group["suffix_rows"].sum())
        base=float(np.sqrt(group["base_sse"].sum()/n))
        candidate=float(np.sqrt(group["candidate_sse"].sum()/n))
        rows.append({"fold":int(fold),"base_rmse":base,"candidate_rmse":candidate,"delta":candidate-base})
    return {"fold_report":rows,"improved_folds":sum(row["delta"]<0 for row in rows)}


def main(argv: list[str] | None=None) -> None:
    args=build_parser().parse_args(argv)
    config=load_config(args.config)
    seed=int(config.get("seed",42))
    stage16,stage17,public_run=args.stage16b_run.resolve(),args.stage17a_run.resolve(),args.public_oof_run.resolve()
    summary17=json.loads((stage17/"summary.json").read_text(encoding="utf-8"))
    expected_hash=str(config["provenance"]["stage16b_manifest_sha256"])
    if summary17.get("stage16b_manifest_sha256")!=expected_hash:
        raise AssertionError("Stage 17A manifest provenance mismatch")
    output=args.artifact_dir.resolve()/args.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True,exist_ok=True)

    cuts=pd.read_parquet(stage17/"cut_report.parquet")
    eligible=cuts[(cuts["evaluation_role"]=="primary") & cuts["replay_eligible"]].copy()
    calibration_config=dict(config.get("calibration",{}))
    eligible=eligible[
        eligible["cut_index"].astype(int)-eligible["original_public_cut_index"].astype(int)
        >= int(calibration_config.get("minimum_calibration_gap_rows",96))
    ].copy()
    excluded_wells:set[str]=set()
    for run in args.exclude_run:
        path=run.resolve()/"cut_features.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        excluded_wells.update(pd.read_parquet(path,columns=["well_id"])["well_id"].astype(str))
    eligible=eligible[~eligible["well_id"].astype(str).isin(excluded_wells)].copy()
    selected=stable_stratified_sample(eligible,int(config.get("sample",{}).get("cuts_per_stratum",4)))
    if args.limit_cuts is not None:
        selected=selected.head(int(args.limit_cuts)).copy()
    selected=selected.sort_values(["well_id","cut_index"],kind="stable").reset_index(drop=True)
    overlap=sorted(set(selected["well_id"].astype(str)).intersection(excluded_wells))
    if overlap:
        raise AssertionError(f"Discovery well leakage: {overlap[:5]}")

    assignments=pd.read_parquet(stage16/"well_assignments.parquet")
    assignments["well_id"]=assignments["well_id"].astype(str)
    assignments["typewell_fold"]=_typewell_folds(
        assignments,int(config.get("validation",{}).get("n_typewell_folds",5)),seed
    )
    selected=selected.merge(
        assignments[["well_id","spatial_fold","typewell_fold","branch_group_fold"]],
        on="well_id",how="left",validate="many_to_one",
    )
    selected["stage16_fold"]=selected["stage16_fold"].astype(int)
    selected_wells=selected["well_id"].astype(str).unique().tolist()
    public=pd.read_parquet(
        public_run/"base_oof.parquet",
        columns=["well_id","row_index","y_pred"],
        filters=[("well_id","in",selected_wells)],
    )
    public["well_id"]=public["well_id"].astype(str)
    public_by_well={well:frame.sort_values("row_index") for well,frame in public.groupby("well_id",sort=True)}
    train=args.data_dir.resolve()/"train"

    @lru_cache(maxsize=64)
    def load_well(well_id:str)->pd.DataFrame:
        return pd.read_csv(train/f"{well_id}__horizontal_well.csv")

    @lru_cache(maxsize=64)
    def load_typewell(well_id:str)->pd.DataFrame:
        return pd.read_csv(train/f"{well_id}__typewell.csv")

    candidate_config=dict(config.get("candidates",{}))
    router_config=dict(config.get("router",{}))
    cut_rows,candidate_rows,invariance=[],[],[]
    for position,cut in enumerate(selected.itertuples(index=False),1):
        well_id,outer=str(cut.well_id),int(cut.cut_index)
        horizontal,typewell=load_well(well_id),load_typewell(well_id)
        source=public_by_well[well_id]
        original=int(source["row_index"].min())
        if original!=int(cut.original_public_cut_index):
            raise AssertionError(f"{well_id}: public OOF cutoff mismatch")
        source_prediction=source["y_pred"].to_numpy(float)
        inner_cuts=internal_cut_indices(original,outer,calibration_config)
        if len(inner_cuts)<2:
            raise AssertionError(f"{cut.cut_id}: fewer than two internal calibration cuts")
        inner_scores:dict[str,list[float]]={}
        for inner in inner_cuts:
            public_inner=source_prediction[inner-original:]
            candidates=build_candidates(horizontal,typewell,inner,public_inner,candidate_config)
            truth=horizontal["TVT"].to_numpy(float)[inner:outer]
            holdout=outer-inner
            for name,prediction in candidates.items():
                score=float(np.sqrt(np.mean(np.square(prediction[:holdout]-truth))))
                inner_scores.setdefault(name,[]).append(score)
        aggregate={name:float(np.median(values)) for name,values in inner_scores.items()}
        public_outer=source_prediction[outer-original:]
        outer_candidates=build_candidates(horizontal,typewell,outer,public_outer,candidate_config)
        base=outer_candidates[str(router_config.get("base_candidate","top_pf_a130"))]
        selected_name=min(aggregate,key=lambda name:(aggregate[name],name))
        raw=outer_candidates[selected_name]
        candidate=_guarded_route(base,raw,router_config)
        truth=horizontal["TVT"].to_numpy(float)[outer:]
        outer_rmse={}
        for name,prediction in outer_candidates.items():
            rmse=float(np.sqrt(np.mean(np.square(prediction-truth))))
            outer_rmse[name]=rmse
            candidate_rows.append({
                "cut_id":str(cut.cut_id),"well_id":well_id,"candidate":name,
                "inner_rmse":aggregate[name],"outer_rmse":rmse,
                "selected":name==selected_name,
            })
        oracle_name=min(outer_rmse,key=lambda name:(outer_rmse[name],name))
        oracle=outer_candidates[oracle_name]
        cut_rows.append({
            "cut_id":str(cut.cut_id),"well_id":well_id,
            "requested_fraction":float(cut.requested_fraction),
            **{family:int(getattr(cut,family)) for family in FOLD_FAMILIES},
            "suffix_rows":len(truth),"base_sse":float(np.square(base-truth).sum()),
            "candidate_sse":float(np.square(candidate-truth).sum()),
            "raw_router_sse":float(np.square(raw-truth).sum()),
            "oracle_sse":float(np.square(oracle-truth).sum()),
            "selected_candidate":selected_name,"oracle_candidate":oracle_name,
            "inner_cuts":len(inner_cuts),
        })
        if len(invariance)<8:
            changed=horizontal.copy()
            changed.loc[changed.index>=outer,"TVT"]+=9999.0
            changed_candidates=build_candidates(changed,typewell,outer,public_outer,candidate_config)
            invariance.append(all(np.array_equal(outer_candidates[name],changed_candidates[name]) for name in outer_candidates))
        if position%10==0:
            print(f"prefix router {position}/{len(selected)} cuts",flush=True)

    report=pd.DataFrame.from_records(cut_rows)
    candidates=pd.DataFrame.from_records(candidate_rows)
    candidates["inner_rank"]=candidates.groupby("cut_id")["inner_rmse"].rank(method="average")
    candidates["outer_rank"]=candidates.groupby("cut_id")["outer_rmse"].rank(method="average")
    rank_correlation=float(candidates[["inner_rank","outer_rank"]].corr().iloc[0,1])
    if not np.isfinite(rank_correlation):
        rank_correlation=0.0
    metrics=_metrics(report)
    bootstrap=_bootstrap(report,int(config.get("validation",{}).get("bootstrap_resamples",2000)),seed)
    rows=int(report["suffix_rows"].sum())
    raw_rmse=float(np.sqrt(report["raw_router_sse"].sum()/rows))
    oracle_rmse=float(np.sqrt(report["oracle_sse"].sum()/rows))
    base_well=report.groupby("well_id").agg(sse=("base_sse","sum"),rows=("suffix_rows","sum"))
    candidate_well=report.groupby("well_id").agg(sse=("candidate_sse","sum"),rows=("suffix_rows","sum"))
    p90_delta=float(
        np.sqrt(candidate_well.sse/candidate_well.rows).quantile(.9)
        -np.sqrt(base_well.sse/base_well.rows).quantile(.9)
    )
    family_reports={family:_family_report(report,family) for family in FOLD_FAMILIES}
    validation=dict(config.get("validation",{}))
    gates={
        "hidden_target_invariance":bool(invariance) and all(invariance),
        "public_oof_target_safe":True,
        "discovery_well_overlap_zero":len(overlap)==0,
        "prefix_outer_rank_correlation":rank_correlation>=float(validation.get("minimum_rank_correlation",.10)),
        "guarded_router_gain":metrics["rmse_delta"]<=-float(validation.get("minimum_gain",.05)),
        "bootstrap_upper_below_zero":bootstrap[1]<0,
        "standard_fold_consistency":metrics["improved_folds"]>=int(validation.get("minimum_improved_folds",4)),
        "fraction_consistency":metrics["improved_fractions"]>=int(validation.get("minimum_improved_fractions",4)),
        "spatial_fold_consistency":family_reports["spatial_fold"]["improved_folds"]>=int(validation.get("minimum_spatial_folds",4)),
        "typewell_fold_consistency":family_reports["typewell_fold"]["improved_folds"]>=int(validation.get("minimum_typewell_folds",4)),
        "branch_group_fold_consistency":family_reports["branch_group_fold"]["improved_folds"]>=int(validation.get("minimum_branch_folds",4)),
        "well_p90_nonworse":p90_delta<=0,
    }
    promoted=bool(all(gates.values()))
    report.to_parquet(output/"router_cut_report.parquet",index=False)
    candidates.to_parquet(output/"candidate_report.parquet",index=False)
    summary={
        "stage21a_complete":True,"promoted_to_stage21b":promoted,
        "stage16b_manifest_sha256":expected_hash,
        "sample_cuts":len(report),"sample_wells":int(report.well_id.nunique()),
        "excluded_discovery_wells":len(excluded_wells),"discovery_well_overlap":overlap,
        "candidate_count":int(candidates.candidate.nunique()),
        "rank_correlation":rank_correlation,
        "top1_oracle_match_fraction":float((report.selected_candidate==report.oracle_candidate).mean()),
        "base_rmse":metrics["base_rmse"],"guarded_router_rmse":metrics["candidate_rmse"],
        "guarded_router_delta":metrics["rmse_delta"],"raw_router_rmse":raw_rmse,
        "oracle_rmse":oracle_rmse,"oracle_delta":oracle_rmse-metrics["base_rmse"],
        "well_p90_delta":p90_delta,"bootstrap_95pct":bootstrap,
        "metrics":metrics,"family_reports":family_reports,"gates":gates,
        "limitations":[
            "well-isolated public OOF substitutes for the unavailable fold-safe learned 470 branch",
            "the frozen 470 visible-prefix overlay is not included in the proxy base",
        ],
        "next_step":(
            "Validate the router at higher PF resolution on a second disjoint-well sample."
            if promoted else
            "Revise the candidate library/calibration rule before any learned router or submission."
        ),
    }
    write_json(output/"summary.json",summary)
    write_json(output/"environment.json",environment_report())
    config["resolved"]={
        "stage16b_run":str(stage16),"stage17a_run":str(stage17),
        "public_oof_run":str(public_run),"exclude_runs":[str(path.resolve()) for path in args.exclude_run],
        "data_dir":str(args.data_dir.resolve()),"run_id":args.run_id,
    }
    write_yaml(output/"config.yaml",config)
    print(summary,flush=True)


if __name__=="__main__":
    main()
