from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def make_group_folds(
    well_stats: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    if "well_id" not in well_stats:
        raise ValueError("well_stats must contain well_id")
    wells = well_stats[["well_id"]].drop_duplicates().sort_values("well_id").reset_index(drop=True)
    if len(wells) < n_splits:
        raise ValueError(f"Need at least {n_splits} wells, got {len(wells)}")

    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(wells), -1, dtype=np.int16)
    placeholder = np.zeros((len(wells), 1), dtype=np.float32)
    groups = wells["well_id"].to_numpy()
    for fold, (_, valid_index) in enumerate(splitter.split(placeholder, groups=groups)):
        folds[valid_index] = fold
    if (folds < 0).any():
        raise RuntimeError("Some wells were not assigned to a fold")
    wells["fold"] = folds
    wells["fold_seed"] = seed
    wells["n_splits"] = n_splits
    return wells


def assert_group_isolation(assignments: pd.DataFrame) -> None:
    fold_counts = assignments.groupby("well_id")["fold"].nunique()
    if not (fold_counts == 1).all():
        raise AssertionError("A well_id appears in more than one fold")

