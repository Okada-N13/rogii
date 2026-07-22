from __future__ import annotations

import pandas as pd

from rogii.cli.selector_resolution import stable_stratified_sample


def test_stratified_sample_is_deterministic_and_bounded() -> None:
    frame = pd.DataFrame({
        "cut_id": [f"c{i}" for i in range(40)],
        "stage16_fold": [i % 2 for i in range(40)],
        "requested_fraction": [0.18 if (i // 2) % 2 == 0 else 0.22 for i in range(40)],
    })
    first = stable_stratified_sample(frame, 3)
    second = stable_stratified_sample(frame.sample(frac=1.0, random_state=1), 3)
    assert first.sort_values("cut_id")["cut_id"].tolist() == second.sort_values("cut_id")["cut_id"].tolist()
    assert len(first) == 12
    assert first.groupby(["stage16_fold", "requested_fraction"]).size().eq(3).all()
