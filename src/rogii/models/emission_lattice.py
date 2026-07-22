from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _logadd(values: np.ndarray, length: int) -> float:
    maximum = -1e300
    for index in range(length):
        if values[index] > maximum:
            maximum = values[index]
    total = 0.0
    for index in range(length):
        total += np.exp(values[index] - maximum)
    return maximum + np.log(total)


@njit(cache=True, nogil=True)
def _forward_backward(
    log_probability: np.ndarray,
    max_jump: int,
    transition_penalty: float,
    zero_penalty: float,
    initial_penalty: float,
) -> np.ndarray:
    steps, states = log_probability.shape
    center = (states - 1) / 2.0
    alpha = np.full((steps, states), -1e300, np.float64)
    beta = np.full((steps, states), -1e300, np.float64)
    scratch = np.empty(2 * max_jump + 1, np.float64)
    for state in range(states):
        distance = state - center
        alpha[0, state] = log_probability[0, state] - initial_penalty * distance * distance
    for step in range(1, steps):
        for state in range(states):
            lower = max(0, state - max_jump)
            upper = min(states - 1, state + max_jump)
            length = 0
            for source in range(lower, upper + 1):
                jump = state - source
                scratch[length] = alpha[step - 1, source] - transition_penalty * jump * jump
                length += 1
            distance = state - center
            alpha[step, state] = (
                log_probability[step, state]
                - zero_penalty * distance * distance
                + _logadd(scratch, length)
            )
    beta[-1, :] = 0.0
    for step in range(steps - 2, -1, -1):
        for state in range(states):
            lower = max(0, state - max_jump)
            upper = min(states - 1, state + max_jump)
            length = 0
            for target in range(lower, upper + 1):
                jump = target - state
                distance = target - center
                scratch[length] = (
                    beta[step + 1, target]
                    + log_probability[step + 1, target]
                    - transition_penalty * jump * jump
                    - zero_penalty * distance * distance
                )
                length += 1
            beta[step, state] = _logadd(scratch, length)
    posterior = np.empty((steps, states), np.float64)
    normalizer_scratch = np.empty(states, np.float64)
    for step in range(steps):
        for state in range(states):
            normalizer_scratch[state] = alpha[step, state] + beta[step, state]
        normalizer = _logadd(normalizer_scratch, states)
        for state in range(states):
            posterior[step, state] = np.exp(normalizer_scratch[state] - normalizer)
    return posterior


@njit(cache=True, nogil=True)
def _viterbi(
    log_probability: np.ndarray,
    max_jump: int,
    transition_penalty: float,
    zero_penalty: float,
    initial_penalty: float,
) -> np.ndarray:
    steps, states = log_probability.shape
    center = (states - 1) / 2.0
    previous = np.empty(states, np.float64)
    back = np.empty((steps, states), np.int16)
    for state in range(states):
        distance = state - center
        previous[state] = -log_probability[0, state] + initial_penalty * distance * distance
        back[0, state] = -1
    for step in range(1, steps):
        current = np.empty(states, np.float64)
        for state in range(states):
            best = 1e300
            best_source = state
            for source in range(max(0, state - max_jump), min(states - 1, state + max_jump) + 1):
                jump = state - source
                cost = previous[source] + transition_penalty * jump * jump
                if cost < best:
                    best = cost
                    best_source = source
            distance = state - center
            current[state] = best - log_probability[step, state] + zero_penalty * distance * distance
            back[step, state] = best_source
        previous = current
    path = np.empty(steps, np.int16)
    path[-1] = int(np.argmin(previous))
    for step in range(steps - 1, 0, -1):
        path[step - 1] = back[step, path[step]]
    return path


@njit(cache=True, nogil=True)
def _k_best_paths(
    log_probability: np.ndarray,
    max_jump: int,
    transition_penalty: float,
    zero_penalty: float,
    initial_penalty: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    steps, states = log_probability.shape
    center = (states - 1) / 2.0
    previous = np.full((states, k), 1e300, np.float64)
    back_state = np.full((steps, states, k), -1, np.int16)
    back_rank = np.full((steps, states, k), -1, np.int16)
    for state in range(states):
        distance = state - center
        previous[state, 0] = -log_probability[0, state] + initial_penalty * distance * distance
    candidates = np.empty((2 * max_jump + 1) * k, np.float64)
    candidate_state = np.empty((2 * max_jump + 1) * k, np.int16)
    candidate_rank = np.empty((2 * max_jump + 1) * k, np.int16)
    for step in range(1, steps):
        current = np.full((states, k), 1e300, np.float64)
        for state in range(states):
            count = 0
            for source in range(max(0, state - max_jump), min(states - 1, state + max_jump) + 1):
                jump = state - source
                for rank in range(k):
                    if previous[source, rank] < 1e299:
                        candidates[count] = previous[source, rank] + transition_penalty * jump * jump
                        candidate_state[count] = source
                        candidate_rank[count] = rank
                        count += 1
            distance = state - center
            local = -log_probability[step, state] + zero_penalty * distance * distance
            for out_rank in range(min(k, count)):
                best_index = 0
                for index in range(1, count):
                    if candidates[index] < candidates[best_index]:
                        best_index = index
                current[state, out_rank] = candidates[best_index] + local
                back_state[step, state, out_rank] = candidate_state[best_index]
                back_rank[step, state, out_rank] = candidate_rank[best_index]
                candidates[best_index] = 1e300
        previous = current
    terminal_cost = previous.ravel().copy()
    paths = np.empty((k, steps), np.int16)
    costs = np.empty(k, np.float64)
    for out_rank in range(k):
        best_index = int(np.argmin(terminal_cost))
        state = best_index // k
        rank = best_index % k
        costs[out_rank] = terminal_cost[best_index]
        terminal_cost[best_index] = 1e300
        paths[out_rank, -1] = state
        for step in range(steps - 1, 0, -1):
            source_state = back_state[step, state, rank]
            source_rank = back_rank[step, state, rank]
            paths[out_rank, step - 1] = source_state
            state, rank = source_state, source_rank
    return paths, costs


def _log_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values -= values.max(axis=1, keepdims=True)
    return values - np.log(np.exp(values).sum(axis=1, keepdims=True))


def decode_lattice(
    logits: np.ndarray,
    offsets: np.ndarray,
    profile: dict[str, Any],
    *,
    k_best: int = 0,
) -> dict[str, np.ndarray | float]:
    log_probability = _log_softmax(logits, float(profile.get("temperature", 1.0)))
    options = (
        int(profile.get("max_jump_states", 3)),
        float(profile.get("transition_penalty", 0.10)),
        float(profile.get("zero_penalty", 0.0002)),
        float(profile.get("initial_penalty", 0.05)),
    )
    posterior = _forward_backward(log_probability, *options)
    posterior_mean = posterior @ np.asarray(offsets, dtype=float)
    viterbi_state = _viterbi(log_probability, *options)
    output: dict[str, np.ndarray | float] = {
        "posterior_mean": posterior_mean.astype(np.float32),
        "posterior_map": np.asarray(offsets)[np.argmax(posterior, axis=1)].astype(np.float32),
        "viterbi": np.asarray(offsets)[viterbi_state].astype(np.float32),
        "posterior_entropy": (-posterior * np.log(np.maximum(posterior, 1e-12))).sum(axis=1).astype(np.float32),
    }
    if k_best > 0:
        paths, costs = _k_best_paths(log_probability, *options, int(k_best))
        weights = np.exp(-(costs - costs.min()) / float(profile.get("path_temperature", 1.0)))
        weights /= weights.sum()
        output["kbest_mean"] = (weights @ np.asarray(offsets)[paths]).astype(np.float32)
        output["kbest_paths"] = np.asarray(offsets)[paths].astype(np.float32)
        output["kbest_weights"] = weights.astype(np.float32)
        output["kbest_effective_paths"] = float(1.0 / np.square(weights).sum())
    return output
