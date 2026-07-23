from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rogii.inference.stage19_trajectory import apply_trajectory_residual
from rogii.models.portable_hgb import export_hist_gradient_boosting, load_portable_hgb, predict_portable_hgb


def test_portable_hgb_is_exact_with_missing_values(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(300, 9))
    x[::11, 2] = np.nan
    y = rng.normal(size=len(x))
    model = HistGradientBoostingRegressor(
        max_iter=20, max_leaf_nodes=7, learning_rate=.04, early_stopping=False
    ).fit(x, y)
    path = tmp_path / "model.npz"
    export_hist_gradient_boosting(model, path)
    assert np.array_equal(model.predict(x), predict_portable_hgb(load_portable_hgb(path), x))


def test_standalone_stage19c_inference_uses_test_inputs(tmp_path: Path) -> None:
    package, test_dir = tmp_path / "package", tmp_path / "data" / "test"
    (package / "models").mkdir(parents=True)
    test_dir.mkdir(parents=True)
    rows, cut = 80, 30
    index = np.arange(rows)
    horizontal = pd.DataFrame({
        "MD": index * 10., "X": 1000.+index*8., "Y": 2000.+index,
        "Z": 900.-index, "GR": 70.+np.sin(index/5),
        "TVT_input": np.where(index < cut, 10000.+index*.2, np.nan),
    })
    typewell = pd.DataFrame({"TVT": np.linspace(9800,10200,200), "GR": 70.+np.sin(np.arange(200)/5)})
    well_id = "abc12345"
    horizontal.to_csv(test_dir / f"{well_id}__horizontal_well.csv", index=False)
    typewell.to_csv(test_dir / f"{well_id}__typewell.csv", index=False)
    feature_columns = ["cut_fraction", "base_boundary_jump", "suffix_gr_mean"]
    x = np.random.default_rng(5).normal(size=(50, len(feature_columns)))
    models = []
    for target_index, target in enumerate(("target_residual_level","target_residual_slope","target_residual_curve")):
        model = HistGradientBoostingRegressor(max_iter=3, min_samples_leaf=2, early_stopping=False).fit(
            x, np.full(len(x), target_index+1.)
        )
        path = package / "models" / f"{target}.npz"
        export_hist_gradient_boosting(model, path)
        import hashlib
        models.append({"seed": 1, "target": target, "file": f"models/{path.name}",
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    inference = Path(__file__).resolve().parents[1] / "src" / "rogii" / "inference" / "stage19_trajectory.py"
    import hashlib
    manifest = {
        "stage19c_trajectory_inference_package": True, "package_version": 2,
        "feature_columns": feature_columns, "coefficient_columns": [m["target"] for m in models],
        "features": {"typewell_shift_grid_ft": [-10,0,10]},
        "profile": {"weight": .5, "cap_ft": 16., "ramp_rows": 30.},
        "models": models, "inference_file": "stage19_trajectory.py",
        "inference_sha256": hashlib.sha256(inference.read_bytes()).hexdigest(),
    }
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    ids = [f"{well_id}_{i}" for i in range(cut, rows)]
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ids, "tvt": 10020.+np.arange(rows-cut)*.2}).to_csv(submission,index=False)
    audit = apply_trajectory_residual(package, tmp_path/"data", submission)
    result = pd.read_csv(submission)
    assert audit["stage19_trajectory_applied"] is True
    assert audit["hidden_target_columns_used"] is False
    assert audit["wells"] == 1 and np.isfinite(result.tvt).all()
