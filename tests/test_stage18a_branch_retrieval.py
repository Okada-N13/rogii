from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.cli.branch_retrieval import _metrics, _weighted_retrieval, stable_sample


def test_weighted_retrieval_prefers_near_gr_match() -> None:
    donor_u = np.array([[10.0, 10.0], [30.0, 30.0]])
    distance = np.array([[10.0, 10.0], [100.0, 100.0]])
    gr_delta = np.array([[1.0, 1.0], [50.0, 50.0]])
    prediction, confidence = _weighted_retrieval(donor_u, distance, gr_delta, 300.0, 30.0)
    assert np.all(prediction < 20.0)
    assert np.all(confidence > 0)


def test_stage18_sample_is_order_independent() -> None:
    frame = pd.DataFrame({
        "cut_id": [f"c{i}" for i in range(40)], "stage16_fold": [i % 2 for i in range(40)],
        "requested_fraction": [0.18 if (i // 2) % 2 == 0 else 0.22 for i in range(40)],
    })
    first = stable_sample(frame, 2)
    second = stable_sample(frame.sample(frac=1.0, random_state=4), 2)
    assert sorted(first["cut_id"]) == sorted(second["cut_id"])
    holdout = stable_sample(frame, 2, offset=2)
    assert set(first["cut_id"]).isdisjoint(holdout["cut_id"])


def test_metrics_reports_fold_and_fraction_consistency() -> None:
    rows = pd.DataFrame({
        "stage16_fold": [0, 1, 0, 1], "requested_fraction": [0.2, 0.2, 0.5, 0.5],
        "suffix_rows": [10, 10, 10, 10], "base_sse": [100.0] * 4,
        "candidate_sse": [64.0] * 4,
    })
    report = _metrics(rows)
    assert report["improved_folds"] == 2
    assert report["improved_fractions"] == 2
    assert all(item["delta"] < 0 for item in report["fraction_report"])
