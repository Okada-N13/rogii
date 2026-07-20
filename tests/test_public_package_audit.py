from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.cli.public_package_audit import _array_summary, _ordered_hash, _tabular_summary


def test_package_audit_verifies_parquet_id_order(tmp_path) -> None:
    frame = pd.DataFrame({"id": ["w_0", "w_1"], "oof_pred": [1.0, 2.0]})
    path = tmp_path / "oof_predictions.parquet"
    frame.to_parquet(path, index=False)
    result = _tabular_summary(path, len(frame), _ordered_hash(frame["id"]))
    assert result["row_count_matches_base"] is True
    assert result["id_order_matches_base"] is True
    assert result["prediction_candidate_columns"] == ["oof_pred"]


def test_package_audit_checks_npy_first_dimension(tmp_path) -> None:
    path = tmp_path / "blend_oof.npy"
    np.save(path, np.zeros((7, 3), dtype=np.float32))
    result = _array_summary(path, base_rows=7)
    assert result["shape"] == [7, 3]
    assert result["first_dimension_matches_base"] is True

