from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.evaluation.metrics import evaluate_predictions, paired_well_bootstrap


def make_predictions(offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b"],
            "MD": [1.0, 2.0, 1.0, 2.0],
            "y_true": [10.0, 11.0, 20.0, 21.0],
            "y_pred": [10.0 + offset, 11.0 + offset, 20.0 + offset, 21.0 + offset],
            "fold": [0, 0, 1, 1],
        }
    )


def test_metrics_for_constant_error() -> None:
    metrics, wells = evaluate_predictions(make_predictions(2.0))
    assert metrics["pooled_rmse"] == 2.0
    assert metrics["mean_bias_rmse"] == 2.0
    assert metrics["slope_error_rmse"] < 1e-12
    assert wells["rmse"].tolist() == [2.0, 2.0]


def test_paired_bootstrap_uses_shared_wells() -> None:
    result = paired_well_bootstrap(make_predictions(1.0), make_predictions(2.0), n_resamples=100, seed=1)
    assert result["n_wells"] == 2
    assert np.isclose(result["mean_well_rmse_delta"], -1.0)
    assert np.isclose(result["ci_2_5"], -1.0)
    assert np.isclose(result["ci_97_5"], -1.0)

