from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.models.safe_cutback import apply_cutback_profile, select_cutback_physics


def _linear_well() -> pd.DataFrame:
    md = np.arange(300, dtype=float)
    z = 1000.0 + 0.2 * md
    surface = 1120.0 + 0.08 * md
    tvt = surface - z
    tvt_input = tvt.copy()
    tvt_input[200:] = np.nan
    return pd.DataFrame({"MD": md, "Z": z, "TVT": tvt, "TVT_input": tvt_input})


def test_cutback_selects_a_valid_physics_path_without_hidden_truth() -> None:
    well = _linear_well()
    candidate, report = select_cutback_physics(
        well,
        cut_fractions=[0.55, 0.70, 0.84],
        degrees=[1, 2],
        tails=[80, 160, None],
    )
    truth = well.loc[well["TVT_input"].isna(), "TVT"].to_numpy(dtype=float)
    assert report["status"] == "ok"
    assert report["best_name"].startswith("poly_u_deg1")
    assert np.sqrt(np.mean(np.square(candidate - truth))) < 1e-8
    perturbed = well.copy()
    perturbed.loc[perturbed["TVT_input"].isna(), "TVT"] += 1000.0
    candidate_perturbed, _ = select_cutback_physics(
        perturbed,
        cut_fractions=[0.55, 0.70, 0.84],
        degrees=[1, 2],
        tails=[80, 160, None],
    )
    np.testing.assert_allclose(candidate_perturbed, candidate)


def test_profile_gate_applies_a_capped_faded_move() -> None:
    base = np.zeros(100)
    physics = np.full(100, 20.0)
    report = {
        "status": "ok",
        "anchor_gain": 3.0,
        "best_score": 5.0,
        "consistency": 1.0,
    }
    profile = {
        "minimum_anchor_gain": 1.0,
        "maximum_best_score": 9.0,
        "minimum_consistency": 0.67,
        "maximum_difference_p95": 55.0,
        "base_alpha": 0.1,
        "gain_alpha": 0.0,
        "maximum_alpha": 0.2,
        "correction_cap": 1.5,
        "fade_tau": 20.0,
    }
    prediction, audit = apply_cutback_profile(base, physics, np.arange(100), report, profile)
    assert audit["applied"] is True
    assert prediction[0] == 0.0
    assert prediction[-1] <= 1.5
    assert prediction[-1] > 1.4


def test_profile_gate_falls_back_to_base_when_cutback_is_weak() -> None:
    base = np.arange(10, dtype=float)
    physics = base + 2.0
    prediction, audit = apply_cutback_profile(
        base,
        physics,
        np.arange(10),
        {"status": "ok", "anchor_gain": 0.1, "best_score": 5.0, "consistency": 1.0},
        {"minimum_anchor_gain": 1.0},
    )
    assert audit["applied"] is False
    np.testing.assert_array_equal(prediction, base)
