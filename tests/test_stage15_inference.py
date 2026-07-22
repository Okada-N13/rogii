from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.data.multicut import TARGET_COLUMNS, build_cut_record, build_inference_record, feature_columns


ROOT = Path(__file__).resolve().parents[1]


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = 40
    md = np.arange(rows, dtype=float) + 10_000.0
    horizontal = pd.DataFrame({
        "well_id": "safe", "row_index": np.arange(rows), "MD": md,
        "X": 100.0 + np.arange(rows), "Y": 200.0 + 0.5 * np.arange(rows),
        "Z": -8000.0 - np.arange(rows), "GR": 80.0 + np.sin(np.arange(rows) / 4.0),
        "TVT": 10_500.0 + 1.2 * np.arange(rows) + 0.01 * np.arange(rows) ** 2,
    })
    typewell = pd.DataFrame({"TVT": np.linspace(10_000, 11_000, 100), "GR": np.linspace(70, 110, 100)})
    return horizontal, typewell


def test_inference_record_matches_training_features_without_suffix_target() -> None:
    horizontal, typewell = _frames()
    cut = 19
    training = build_cut_record(horizontal, typewell, cut)
    inference_frame = horizontal.drop(columns="TVT").copy()
    inference_frame["TVT_input"] = horizontal["TVT"].where(horizontal["row_index"] < cut)
    inference = build_inference_record(inference_frame, typewell)
    columns = feature_columns(pd.DataFrame([training]))
    assert set(TARGET_COLUMNS).isdisjoint(inference)
    assert columns == feature_columns(pd.DataFrame([inference]))
    assert np.allclose([training[name] for name in columns], [inference[name] for name in columns])


def test_inference_record_rejects_noncontiguous_prefix() -> None:
    horizontal, typewell = _frames()
    frame = horizontal.drop(columns="TVT").copy()
    frame["TVT_input"] = horizontal["TVT"].where(horizontal["row_index"] < 20)
    frame.loc[25, "TVT_input"] = 123.0
    try:
        build_inference_record(frame, typewell)
    except ValueError as error:
        assert "contiguous prefix" in str(error)
    else:
        raise AssertionError("noncontiguous TVT_input was accepted")


def test_stage15_notebooks_are_self_contained_and_compile() -> None:
    for name in ("330_build_stage15_fold_safe_package.ipynb", "340_kaggle_stage15_internet_off_inference.ipynb"):
        payload = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        code = "\n\n".join("".join(cell.get("source", [])) for cell in payload["cells"] if cell["cell_type"] == "code")
        compile(code, name, "exec")
    kaggle = json.loads((ROOT / "notebooks" / "340_kaggle_stage15_internet_off_inference.ipynb").read_text(encoding="utf-8"))
    assert kaggle["metadata"]["stage15"]["internet_off"] is True
    assert kaggle["metadata"]["stage15"]["single_submission_csv"] is True
