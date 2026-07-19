from __future__ import annotations

import pandas as pd

from rogii.data.folds import assert_group_isolation, make_group_folds


def test_group_folds_are_fixed_and_isolated() -> None:
    stats = pd.DataFrame({"well_id": [f"well_{index:02d}" for index in range(20)]})
    first = make_group_folds(stats, n_splits=5, seed=42)
    second = make_group_folds(stats.sample(frac=1.0, random_state=7), n_splits=5, seed=42)

    assert_group_isolation(first)
    assert first.sort_values("well_id").reset_index(drop=True).equals(
        second.sort_values("well_id").reset_index(drop=True)
    )
    assert sorted(first["fold"].unique()) == [0, 1, 2, 3, 4]

