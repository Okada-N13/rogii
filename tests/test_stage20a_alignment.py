from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rogii.cli.trajectory_base_alignment import main, robust_projection, top_pf_proxy


ROOT = Path(__file__).resolve().parents[1]


def _horizontal(rows: int = 80) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame({
        "MD": index * 10.0, "X": index * 8.0, "Y": index * 2.0,
        "Z": 900.0-index, "GR": 70.0+np.sin(index/5.0),
        "TVT": 10000.0+index*.2,
    })


def test_top_pf_proxy_ignores_hidden_target_and_is_finite() -> None:
    horizontal = _horizontal()
    cut = 30
    public = np.linspace(10006.0, 10018.0, len(horizontal)-cut)
    selector = public + 2.0*np.sin(np.arange(len(public))/7.0)
    config = {
        "ridge_weight": .3, "selector_weight": .7,
        "projection_degree": 3, "projection_blend_weight": .75,
        "final_sp45_weight": .6,
    }
    first = top_pf_proxy(horizontal, cut, public, selector, config)
    changed = horizontal.copy()
    changed.loc[cut:, "TVT"] += 99999.0
    second = top_pf_proxy(changed, cut, public, selector, config)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all() and len(first) == len(public)
    projected = robust_projection(horizontal, cut, .3*public+.7*selector)
    np.testing.assert_allclose(first, .6*projected+.4*public)


def test_stage20a_notebook_is_clean_standalone_and_compiles() -> None:
    path = ROOT / "notebooks" / "520_run_stage20a_top_pf_alignment.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "rogii-trajectory-base-alignment" in text
    assert "stage17_public_replay_full_v002" in text
    assert payload["metadata"]["stage20a"]["submission"] is False
    assert payload["metadata"]["stage20a"]["a130_gr_sigma_multiplier"] == 1.3
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")


def test_stage20a_cli_writes_alignment_artifacts(tmp_path: Path) -> None:
    data, artifacts = tmp_path/"data", tmp_path/"artifacts"
    train, stage16, stage17 = data/"train", artifacts/"stage16", artifacts/"stage17"
    for path in (train,stage16,stage17):
        path.mkdir(parents=True)
    assignments, cuts, predictions = [], [], []
    for well_index in range(15):
        well_id = f"w{well_index:07d}"
        horizontal = _horizontal(60)
        horizontal["X"] += well_index*100
        horizontal["Y"] += well_index*50
        horizontal["GR"] += well_index*.2
        horizontal["TVT"] += well_index
        horizontal["TVT_input"] = np.where(np.arange(60)<20,horizontal["TVT"],np.nan)
        typewell = pd.DataFrame({
            "TVT": np.linspace(9900,10200,150),
            "GR": 70+np.sin(np.arange(150)/5)+well_index*.2,
        })
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
        cut_id=f"{well_id}__cut20"; truth=horizontal.TVT.to_numpy(float)[20:]
        cuts.append({
            "cut_id":cut_id,"well_id":well_id,"cut_index":20,"requested_fraction":1/3,
            "evaluation_role":"primary","replay_eligible":True,"stage16_fold":well_index%5,
            "suffix_rows":40,
        })
        predictions.extend({
            "cut_id":cut_id,"row_index":row,"y_pred":float(value-1.5)
        } for row,value in zip(range(20,60),truth,strict=True))
    pd.DataFrame(assignments).to_parquet(stage16/"well_assignments.parquet",index=False)
    pd.DataFrame(cuts).to_parquet(stage17/"cut_report.parquet",index=False)
    pd.DataFrame(predictions).to_parquet(stage17/"replay_predictions.parquet",index=False)
    (stage17/"summary.json").write_text(json.dumps({"stage16b_manifest_sha256":"synthetic"}),encoding="utf-8")
    config = {
        "seed":42,"provenance":{"stage16b_manifest_sha256":"synthetic"},
        "sample":{"cuts_per_stratum":3},
        "selector":{"particles":4,"seeds":1,"maximum_tracking_steps":10,"gr_sigma_multiplier":1.3},
        "top_pf_proxy":{"ridge_weight":.3,"selector_weight":.7,"projection_degree":2,
                        "projection_blend_weight":.75,"final_sp45_weight":.6},
        "features":{"typewell_shift_grid_ft":[-10,0,10]},
        "target":{"ramp_rows":20,"ridge":1},
        "model":{"max_iter":5,"max_leaf_nodes":3,"max_depth":2,"min_samples_leaf":2},
        "profile":{"weight":.1,"cap_ft":8,"ramp_rows":20},
        "diagnostics":{"weights":[.05,.1]},
        "validation":{"n_typewell_folds":5,"bootstrap_resamples":20,
                      "minimum_standard_gain":0,"minimum_improved_fold_fraction":0},
    }
    config_path=tmp_path/"stage20.yaml"
    config_path.write_text(yaml.safe_dump(config),encoding="utf-8")
    main([
        "--config",str(config_path),"--stage16b-run",str(stage16),"--stage17a-run",str(stage17),
        "--data-dir",str(data),"--artifact-dir",str(artifacts),"--run-id","stage20-test",
    ])
    run=artifacts/"stage20-test"
    summary=json.loads((run/"summary.json").read_text(encoding="utf-8"))
    assert summary["stage20a_complete"] is True
    assert summary["sample_cuts"] == 15
    assert summary["sample_wells"] == 15
    assert summary["gates"]["hidden_target_invariance"] is True
    for name in ["base_comparison.parquet","top_pf_proxy_predictions.parquet",
                 "cut_features.parquet","weight_report.parquet","standard_cut_metrics.parquet"]:
        assert (run/name).is_file(),name
