from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.models.particle_filter import (
    _robust_gr_calibration,
    _weighted_two_mode_hedge,
    predict_particle_filter,
)


def make_pf_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    typewell_tvt = np.arange(90.0, 121.0, 0.25)
    typewell_gr = 100.0 + 25.0 * np.sin((typewell_tvt - 90.0) / 2.5)
    typewell = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})

    md = np.arange(40, dtype=float)
    tvt = 100.0 + 0.08 * md
    gr = np.interp(tvt, typewell_tvt, typewell_gr)
    tvt_input = tvt.copy()
    tvt_input[20:] = np.nan
    horizontal = pd.DataFrame(
        {
            "well_id": ["00abc123"] * len(md),
            "row_index": np.arange(len(md)),
            "MD": md,
            "X": 1.0,
            "Y": 2.0,
            "Z": np.zeros(len(md)),
            "GR": gr,
            "TVT": tvt,
            "TVT_input": tvt_input,
        }
    )
    return horizontal, typewell


def pf_config() -> dict[str, object]:
    return {
        "seed": 42,
        "n_particles": 64,
        "n_seeds": 4,
        "tracking_stride": 1,
        "grid_step": 0.2,
        "likelihood_scale": 5.0,
        "initial_spread": 1.0,
        "momentum": 0.998,
        "rate_noise": 0.001,
        "position_noise": 0.002,
        "rough_position": 0.05,
        "rough_rate": 0.0005,
        "resample_threshold": 0.5,
        "hold_weight": 0.0,
    }


def test_pf_is_deterministic_and_finite() -> None:
    horizontal, typewell = make_pf_inputs()
    first = predict_particle_filter(horizontal, typewell, pf_config())
    second = predict_particle_filter(horizontal, typewell, pf_config())
    assert len(first) == 20
    assert np.isfinite(first["y_pred"]).all()
    assert np.isfinite(first["pf_seed_std"]).all()
    assert np.allclose(first["y_pred"], second["y_pred"])


def test_pf_does_not_read_hidden_tvt_targets() -> None:
    horizontal, typewell = make_pf_inputs()
    original = predict_particle_filter(horizontal, typewell, pf_config())
    modified = horizontal.copy()
    modified.loc[modified["TVT_input"].isna(), "TVT"] += 999.0
    changed = predict_particle_filter(modified, typewell, pf_config())
    assert np.allclose(original["y_pred"], changed["y_pred"])
    assert not np.allclose(original["y_true"], changed["y_true"])


def test_seed_batches_average_independent_pf_predictions() -> None:
    horizontal, typewell = make_pf_inputs()
    first_config = pf_config()
    first_config["seed"] = 42
    second_config = pf_config()
    second_config["seed"] = 43
    ensemble_config = pf_config()
    ensemble_config["seed_batches"] = [42, 43]
    first = predict_particle_filter(horizontal, typewell, first_config)
    second = predict_particle_filter(horizontal, typewell, second_config)
    ensemble = predict_particle_filter(horizontal, typewell, ensemble_config)
    expected = 0.5 * (first["y_pred"].to_numpy() + second["y_pred"].to_numpy())
    assert np.allclose(ensemble["y_pred"], expected)


def test_heel_calibration_recovers_gain_and_offset() -> None:
    reference = np.linspace(40.0, 160.0, 200)
    observed = 1.6 * reference + 22.0
    observed[::31] += 80.0
    gain, offset, calibrated = _robust_gr_calibration(
        observed,
        reference,
        {"minimum_points": 40, "gain_min": 0.5, "gain_max": 2.0, "offset_limit": 100.0},
    )
    assert calibrated
    assert np.isclose(gain, 1.6, atol=0.03)
    assert np.isclose(offset, 22.0, atol=3.0)


def test_bimodal_hedge_is_gated_and_direction_free() -> None:
    levels = np.r_[np.full(7, 100.0), np.full(5, 112.0)]
    weights = np.r_[np.full(7, 0.08), np.full(5, 0.088)]
    shift, separation, minor_mass = _weighted_two_mode_hedge(
        levels,
        weights,
        {"minimum_mass": 0.22, "minimum_separation": 4.0, "maximum_separation": 40.0, "alpha": 1.4, "cap": 4.0},
    )
    assert np.isclose(separation, 12.0)
    assert minor_mass > 0.22
    assert 0.0 < shift <= 4.0


def test_multitemperature_geometry_selector_is_deterministic() -> None:
    horizontal, typewell = make_pf_inputs()
    config = pf_config()
    config.update(
        {
            "likelihood_scales": [3.0, 5.0, 8.0, 12.0],
            "heel_calibration": {"enabled": True, "minimum_points": 10},
            "mha": {"enabled": True},
            "geometry_selector": {
                "enabled": True,
                "n_eval_threshold": 10,
                "z_span_thresholds": [1.0, 2.0],
                "variants": {"1": {"likelihood_scale": 3.0, "hold_weight": 0.15}},
                "default": {"likelihood_scale": 8.0, "hold_weight": 0.2},
            },
        }
    )
    first = predict_particle_filter(horizontal, typewell, config)
    second = predict_particle_filter(horizontal, typewell, config)
    assert np.allclose(first["y_pred"], second["y_pred"])
    assert (first["pf_selected_likelihood_scale"] == 3.0).all()
    assert (first["pf_selector_code"] == 1).all()
    assert np.isfinite(first[["pf_gr_gain", "pf_gr_offset", "pf_mha_shift"]]).all().all()
