from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.evaluation.delta_u_gate import (
    absolute_tail_metrics,
    nested_select_predictions,
    prediction_report,
    select_robust_inference_spec,
)


def _base_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for well in range(15):
        for row in range(8):
            truth = float(well + row * 0.1)
            rows.append(
                {
                    "id": f"w{well}_{row}",
                    "well_id": f"w{well}",
                    "MD": float(row),
                    "y_true": truth,
                    "y_pred": truth + 4.0 + (well % 3),
                    "fold": well % 3,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_absolute_tail_reports_sse_not_only_share() -> None:
    base = _base_frame()
    tail = absolute_tail_metrics(base)
    assert tail["tail_wells"] == 2
    assert tail["worst_tail_sse"] > 0.0
    assert 0.0 < tail["worst_tail_sse_share"] <= 1.0
    assert tail["well_rmse_cvar"] >= tail["well_rmse_p90"]


def test_nested_selection_never_uses_outer_fold() -> None:
    base = _base_frame()
    truth = base["y_true"].to_numpy(dtype=float)
    predictions = {
        "good": truth + 1.0,
        "bad": base["y_pred"].to_numpy(dtype=float) + 2.0,
    }
    nested, selections = nested_select_predictions(
        base,
        predictions,
        {
            "minimum_selection_gain": 0.1,
            "inner_fold_tolerance": 0.0,
            "tail_tolerance": 0.0,
        },
    )
    assert [row["selected_spec"] for row in selections] == ["good", "good", "good"]
    np.testing.assert_allclose(nested["y_pred"], truth + 1.0)


def test_robust_inference_spec_requires_every_family() -> None:
    base = _base_frame()
    truth = base["y_true"].to_numpy(dtype=float)
    good = prediction_report(base, truth + 1.0)
    weaker = prediction_report(base, truth + 2.0)
    bad = prediction_report(base, base["y_pred"].to_numpy(dtype=float) + 1.0)
    reports = {
        "fold": {"good": good, "weaker": weaker, "bad": bad},
        "spatial_fold": {"good": good, "weaker": weaker, "bad": bad},
        "typewell_fold": {"good": good, "weaker": weaker, "bad": bad},
    }
    selected, rows = select_robust_inference_spec(
        reports, {"inference_fold_tolerance": 0.0, "tail_tolerance": 0.0}
    )
    assert selected == "good"
    assert sum(row["eligible"] for row in rows) == 2
