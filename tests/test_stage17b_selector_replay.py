from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.cli.selector_replay import (
    _sample_positions,
    likelihood_selector,
    parse_selector_variant,
    selector_variant,
)


def test_selector_variant_matches_frozen_v599_bins() -> None:
    code, name = selector_variant(1000, np.array([0.0, 50.0]))
    assert code == 0 and name == "pf_scale_5_hold_0.2"
    code, name = selector_variant(6000, np.array([0.0, 200.0]))
    assert code == 5 and name == "pf_scale_12_beam_0.2_hold_0.05"
    assert parse_selector_variant(name) == (12.0, 0.2, 0.05)


def test_tracking_subsample_preserves_endpoints() -> None:
    positions = _sample_positions(5000, 512)
    assert len(positions) <= 512
    assert positions[0] == 0 and positions[-1] == 4999
    assert np.all(np.diff(positions) > 0)


def test_selector_does_not_use_hidden_suffix_tvt() -> None:
    rows = 40
    horizontal = pd.DataFrame({
        "MD": np.arange(rows, dtype=float), "Z": np.linspace(0.0, 10.0, rows),
        "GR": 50.0 + np.sin(np.arange(rows) / 4.0),
        "TVT": 1000.0 + np.arange(rows) * 0.2,
    })
    typewell = pd.DataFrame({
        "TVT": np.linspace(950.0, 1100.0, 100),
        "GR": 50.0 + np.sin(np.arange(100) / 10.0),
    })
    config = {"particles": 8, "seeds": 2, "maximum_tracking_steps": 20, "seed_base": 0}
    first, _ = likelihood_selector(horizontal, typewell, 15, config)
    changed = horizontal.copy()
    changed.loc[15:, "TVT"] += 10000.0
    second, _ = likelihood_selector(changed, typewell, 15, config)
    np.testing.assert_array_equal(first, second)


def test_a130_multiplier_changes_only_visible_likelihood_width() -> None:
    rows = 50
    horizontal = pd.DataFrame({
        "MD": np.arange(rows, dtype=float), "Z": np.linspace(0.0, 12.0, rows),
        "GR": 50.0 + 15.0 * np.sin(np.arange(rows) / 5.0),
        "TVT": 1000.0 + np.arange(rows) * 0.2,
    })
    typewell = pd.DataFrame({
        "TVT": np.linspace(950.0, 1100.0, 120),
        "GR": 50.0 + 8.0 * np.sin(np.arange(120) / 9.0),
    })
    config = {"particles": 12, "seeds": 3, "maximum_tracking_steps": 25, "seed_base": 0}
    _, ordinary = likelihood_selector(horizontal, typewell, 20, config)
    _, a130 = likelihood_selector(horizontal, typewell, 20, {**config, "gr_sigma_multiplier": 1.3})
    assert np.isclose(a130["gr_sigma"], ordinary["gr_sigma"] * 1.3)
