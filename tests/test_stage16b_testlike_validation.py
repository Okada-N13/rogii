from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.cli.testlike_validation import _pseudo_manifest


def _well(well_id: str, rows: int = 200) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame({
        "well_id": well_id, "row_index": index, "MD": 10_000.0 + index,
        "X": 1000.0 + index, "Y": 2000.0 + index * 0.5, "Z": -8000.0 - index,
        "GR": 80.0 + np.sin(index / 7.0), "TVT": 10_500.0 + index * 0.2,
    })


def test_manifest_uses_short_prefix_and_never_exposes_target() -> None:
    manifest = _pseudo_manifest(
        {"a": _well("a"), "b": _well("b")},
        {
            "fractions": [0.18, 0.22, 0.26, 0.30, 0.34],
            "diagnostic_fractions": [0.50],
            "min_prefix_rows": 20, "min_suffix_rows": 20,
        },
    )
    primary = manifest[manifest["evaluation_role"] == "primary"]
    assert len(primary) == 10
    assert primary["cut_fraction"].between(0.17, 0.35).all()
    assert not manifest["target_visible_to_features"].any()
    assert manifest["visible_prefix_sha256"].str.len().eq(64).all()
    assert not manifest["cut_id"].duplicated().any()
