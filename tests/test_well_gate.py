from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.models.well_gate import WELL_GATE_FEATURES, build_well_gate_table


def _matrix(truth_shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for well_number, well in enumerate(["a", "b"]):
        for row in range(20):
            base = 100.0 + well_number + 0.1 * row
            rows.append(
                {
                    "id": f"{well}_{row}",
                    "well_id": well,
                    "row_index": row,
                    "evaluation_md": float(row),
                    "fold": well_number,
                    "spatial_fold": well_number,
                    "base_y_pred": base,
                    "physics_y_pred": base + 2.0 + 0.02 * row,
                    "pred_conservative": base + 0.2,
                    "y_true": base + truth_shift + (0.2 if well == "a" else -0.2),
                }
            )
    return pd.DataFrame(rows)


def _reports() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "well_id": well,
                "best_name": "poly_u_deg2_tail160",
                "known_rows": 100,
                "hidden_rows": 20,
                "best_score": 4.0,
                "anchor_score": 6.0,
                "anchor_gain": 2.0,
                "consistency": 2 / 3,
            }
            for well in ["a", "b"]
        ]
    )


def test_well_gate_features_exclude_target_derived_values() -> None:
    table = build_well_gate_table(_matrix(), _reports())
    changed = build_well_gate_table(_matrix(truth_shift=50.0), _reports())
    np.testing.assert_allclose(
        table[WELL_GATE_FEATURES].to_numpy(dtype=float),
        changed[WELL_GATE_FEATURES].to_numpy(dtype=float),
    )
    assert not any("target" in name or "true" in name for name in WELL_GATE_FEATURES)
    assert not np.allclose(table["gain_target"], changed["gain_target"])


def test_well_gate_table_parses_physics_spec() -> None:
    table = build_well_gate_table(_matrix(), _reports())
    assert table["degree"].eq(2.0).all()
    assert table["tail"].eq(160.0).all()
    assert table["hidden_to_known_ratio"].eq(0.2).all()
