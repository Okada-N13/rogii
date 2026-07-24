from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rogii.cli.prefix_router import build_candidates, internal_cut_indices, main


ROOT=Path(__file__).resolve().parents[1]


def _frames(rows:int=180,well_id:str="w0000000")->tuple[pd.DataFrame,pd.DataFrame]:
    index=np.arange(rows)
    md=index*10.0
    horizontal=pd.DataFrame({
        "MD":md,"X":1000+md,"Y":2000+md*.1,"Z":900-md*.08,
        "GR":70+8*np.sin(index/6),"TVT":10000+index*.3,
    })
    typewell=pd.DataFrame({
        "TVT":np.linspace(9800,10400,300),
        "GR":70+8*np.sin(np.arange(300)/10),
    })
    return horizontal,typewell


def test_candidate_library_and_internal_cuts_are_target_safe() -> None:
    horizontal,typewell=_frames()
    cut=120
    public=horizontal.TVT.to_numpy(float)[cut:]-1
    config={
        "gr_sigma_multipliers":[1.3],"polynomial_degrees":[1],
        "selector":{"particles":4,"seeds":1,"maximum_tracking_steps":12},
        "top_pf_proxy":{"ridge_weight":.3,"selector_weight":.7,"projection_degree":2,
                        "projection_blend_weight":.75,"final_sp45_weight":.6},
    }
    first=build_candidates(horizontal,typewell,cut,public,config)
    changed=horizontal.copy(); changed.loc[cut:,"TVT"]+=9999
    second=build_candidates(changed,typewell,cut,public,config)
    assert set(first)=={"public_oof","selector_a130","top_pf_a130","poly_u_deg1"}
    for name in first:
        np.testing.assert_array_equal(first[name],second[name])
    assert len(internal_cut_indices(20,120,{"inner_gap_fractions":[.45,.72],"minimum_holdout_rows":10,
                                                   "minimum_calibration_gap_rows":90}))==2


def test_stage21a_notebook_is_clean_and_compiles() -> None:
    path=ROOT/"notebooks"/"540_run_stage21a_prefix_router.ipynb"
    payload=json.loads(path.read_text(encoding="utf-8"))
    text="\n".join("".join(cell.get("source",[])) for cell in payload["cells"])
    assert "rogii-prefix-router" in text and text.count("'--exclude-run'")==2
    assert payload["metadata"]["stage21a"]["submission"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type")=="code":
            assert cell["execution_count"] is None and cell["outputs"]==[]
            compile("".join(cell.get("source",[])),str(path),"exec")


def test_stage21a_cli_writes_router_artifacts(tmp_path:Path) -> None:
    data,artifacts=tmp_path/"data",tmp_path/"artifacts"
    train,stage16,stage17,public_run=data/"train",artifacts/"stage16",artifacts/"stage17",artifacts/"public"
    for path in (train,stage16,stage17,public_run):
        path.mkdir(parents=True)
    assignments,cuts,public_rows=[],[],[]
    for well_index in range(15):
        well_id=f"w{well_index:07d}"
        horizontal,typewell=_frames(well_id=well_id)
        horizontal["X"]+=well_index*100; horizontal["Y"]+=well_index*50
        horizontal["TVT"]+=well_index; horizontal["GR"]+=well_index*.1
        typewell["GR"]+=well_index*.1
        horizontal["TVT_input"]=np.where(np.arange(len(horizontal))<20,horizontal.TVT,np.nan)
        horizontal.to_csv(train/f"{well_id}__horizontal_well.csv",index=False)
        typewell.to_csv(train/f"{well_id}__typewell.csv",index=False)
        assignments.append({
            "well_id":well_id,"fold":well_index%5,"spatial_fold":(well_index//3)%5,
            "branch_group_fold":(well_index*2)%5,
            "typewell_gr_mean":float(typewell.GR.mean()),"typewell_gr_std":float(typewell.GR.std()),
            "typewell_gr_q10":float(typewell.GR.quantile(.1)),"typewell_gr_q50":float(typewell.GR.quantile(.5)),
            "typewell_gr_q90":float(typewell.GR.quantile(.9)),"typewell_tvt_min":float(typewell.TVT.min()),
            "typewell_tvt_max":float(typewell.TVT.max()),"typewell_tvt_span":float(typewell.TVT.max()-typewell.TVT.min()),
        })
        cut_id=f"{well_id}__cut120"
        cuts.append({
            "cut_id":cut_id,"well_id":well_id,"cut_index":120,"original_public_cut_index":20,
            "requested_fraction":2/3,"evaluation_role":"primary","replay_eligible":True,
            "stage16_fold":well_index%5,"suffix_rows":60,
        })
        public_rows.extend({
            "well_id":well_id,"row_index":row,"y_pred":float(value-1)
        } for row,value in zip(range(20,180),horizontal.TVT.to_numpy(float)[20:],strict=True))
    pd.DataFrame(assignments).to_parquet(stage16/"well_assignments.parquet",index=False)
    pd.DataFrame(cuts).to_parquet(stage17/"cut_report.parquet",index=False)
    pd.DataFrame(public_rows).to_parquet(public_run/"base_oof.parquet",index=False)
    (stage17/"summary.json").write_text(json.dumps({"stage16b_manifest_sha256":"synthetic"}),encoding="utf-8")
    config={
        "seed":42,"provenance":{"stage16b_manifest_sha256":"synthetic"},
        "sample":{"cuts_per_stratum":3},
        "calibration":{"minimum_calibration_gap_rows":90,"minimum_holdout_rows":10,
                       "inner_gap_fractions":[.45,.72]},
        "candidates":{"gr_sigma_multipliers":[1.3],"polynomial_degrees":[1],
                      "selector":{"particles":3,"seeds":1,"maximum_tracking_steps":8},
                      "top_pf_proxy":{"ridge_weight":.3,"selector_weight":.7,"projection_degree":2,
                                      "projection_blend_weight":.75,"final_sp45_weight":.6}},
        "router":{"base_candidate":"top_pf_a130","blend_weight":.25,"cap_ft":12,"ramp_rows":20},
        "validation":{"n_typewell_folds":5,"bootstrap_resamples":20,"minimum_rank_correlation":-1,
                      "minimum_gain":-999,"minimum_improved_folds":0,"minimum_improved_fractions":0,
                      "minimum_spatial_folds":0,"minimum_typewell_folds":0,"minimum_branch_folds":0},
    }
    config_path=tmp_path/"stage21.yaml"; config_path.write_text(yaml.safe_dump(config),encoding="utf-8")
    main([
        "--config",str(config_path),"--stage16b-run",str(stage16),"--stage17a-run",str(stage17),
        "--public-oof-run",str(public_run),"--data-dir",str(data),
        "--artifact-dir",str(artifacts),"--run-id","stage21-test",
    ])
    run=artifacts/"stage21-test"
    summary=json.loads((run/"summary.json").read_text(encoding="utf-8"))
    assert summary["stage21a_complete"] is True
    assert summary["sample_cuts"]==15 and summary["candidate_count"]==4
    assert summary["gates"]["hidden_target_invariance"] is True
    assert (run/"router_cut_report.parquet").is_file()
    assert (run/"candidate_report.parquet").is_file()
