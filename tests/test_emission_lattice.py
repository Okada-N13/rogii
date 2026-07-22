from __future__ import annotations

import numpy as np

from rogii.models.emission_lattice import decode_lattice


def test_lattice_recovers_smooth_high_probability_path() -> None:
    offsets = np.arange(-10, 11, 2, dtype=np.float32)
    states = np.array([5, 5, 6, 6, 7, 7, 6, 6, 5, 5])
    logits = np.full((len(states), len(offsets)), -4.0, dtype=np.float32)
    logits[np.arange(len(states)), states] = 4.0
    result = decode_lattice(
        logits,
        offsets,
        {"max_jump_states": 2, "transition_penalty": 0.1, "zero_penalty": 0.0, "initial_penalty": 0.0},
        k_best=4,
    )
    np.testing.assert_allclose(result["viterbi"], offsets[states])
    np.testing.assert_allclose(result["posterior_mean"], offsets[states], atol=0.1)
    assert np.asarray(result["kbest_paths"]).shape == (4, len(states))
    assert np.isclose(np.asarray(result["kbest_weights"]).sum(), 1.0)


def test_lattice_smooths_an_isolated_wrong_emission() -> None:
    offsets = np.arange(-10, 11, 2, dtype=np.float32)
    logits = np.full((9, len(offsets)), -3.0, dtype=np.float32)
    logits[:, 5] = 3.0
    logits[4, 10] = 5.0
    result = decode_lattice(
        logits,
        offsets,
        {"max_jump_states": 1, "transition_penalty": 1.0, "zero_penalty": 0.0, "initial_penalty": 0.0},
    )
    assert float(np.asarray(result["viterbi"])[4]) == 0.0
