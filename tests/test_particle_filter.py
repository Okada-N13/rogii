from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.models.particle_filter import predict_particle_filter


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
