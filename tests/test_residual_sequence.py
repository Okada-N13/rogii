from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.models.residual_sequence import build_residual_features


def _write_synthetic_well(root: Path) -> tuple[Path, pd.DataFrame]:
    well_id = "00abc123"
    md = np.arange(80, dtype=float)
    z = -1000.0 - 0.9 * md
    surface = 100.0 + 0.03 * md
    tvt = surface - z
    gr = 100.0 + 20.0 * np.sin(tvt / 4.0)
    tvt_input = tvt.copy()
    tvt_input[40:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": md,
            "X": 1.0,
            "Y": 2.0,
            "Z": z,
            "TVT": tvt,
            "GR": gr,
            "TVT_input": tvt_input,
        }
    )
    path = root / f"{well_id}__horizontal_well.csv"
    horizontal.to_csv(path, index=False)
    typewell_tvt = np.arange(tvt.min() - 10.0, tvt.max() + 10.0, 0.25)
    pd.DataFrame(
        {"TVT": typewell_tvt, "GR": 100.0 + 20.0 * np.sin(typewell_tvt / 4.0)}
    ).to_csv(root / f"{well_id}__typewell.csv", index=False)
    return path, horizontal


def _base_oof(horizontal: pd.DataFrame) -> pd.DataFrame:
    target = horizontal.iloc[40:]
    prediction = target["TVT"].to_numpy(dtype=float) + 2.0
    return pd.DataFrame(
        {
            "id": "00abc123_" + target.index.astype(str),
            "well_id": "00abc123",
            "row_index": target.index,
            "MD": target["MD"].to_numpy(),
            "Z": target["Z"].to_numpy(),
            "anchor_value": float(horizontal["TVT_input"].dropna().iloc[-1]),
            "surface_anchor": float(
                horizontal["TVT_input"].dropna().iloc[-1] + horizontal["Z"].iloc[39]
            ),
            "pf_seed_std": 1.0,
            "pf_gr_sigma": 20.0,
            "pf_log_likelihood_spread": 3.0,
            "y_pred": prediction,
            "y_true": target["TVT"].to_numpy(),
            "fold": 0,
        }
    )


def test_residual_features_do_not_read_hidden_targets(tmp_path: Path) -> None:
    path, horizontal = _write_synthetic_well(tmp_path)
    base = _base_oof(horizontal)
    original = build_residual_features(base, {"00abc123": path}, [9, 33], 16)

    modified_horizontal = horizontal.copy()
    modified_horizontal.loc[40:, "TVT"] += 999.0
    modified_horizontal.to_csv(path, index=False)
    modified_base = base.copy()
    modified_base["y_true"] += 999.0
    changed = build_residual_features(modified_base, {"00abc123": path}, [9, 33], 16)

    assert original.columns == changed.columns
    assert np.allclose(
        original.frame[original.columns].to_numpy(),
        changed.frame[changed.columns].to_numpy(),
    )
    assert not np.allclose(original.frame["residual_target"], changed.frame["residual_target"])
