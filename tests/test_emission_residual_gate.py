from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.cli.emission_residual_gate import _absolute_tail_safe, _candidate_predictions


def test_extended_candidate_uses_saved_crossfit_residuals() -> None:
    frame = pd.DataFrame(
        {
            "y_pred": [10.0, 20.0],
            "raw_generic_residual": [20.0, -4.0],
            "raw_stacked_residual": [2.0, -20.0],
        }
    )
    predictions = _candidate_predictions(frame, [{"name": "w050_cap8", "weight": 0.5, "cap": 8.0}])
    np.testing.assert_allclose(predictions["generic_w050_cap8"], [14.0, 18.0])
    np.testing.assert_allclose(predictions["stacked_w050_cap8"], [11.0, 16.0])


def test_absolute_tail_gate_ignores_share_when_absolute_losses_improve() -> None:
    report = {
        "base_tail": {"worst_tail_sse": 100.0, "well_rmse_cvar": 20.0, "well_rmse_p90": 15.0, "well_rmse_max": 50.0},
        "candidate_tail": {"worst_tail_sse": 95.0, "well_rmse_cvar": 19.0, "well_rmse_p90": 14.0, "well_rmse_max": 48.0},
    }
    assert _absolute_tail_safe(report, 0.0)
