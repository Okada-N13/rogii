from __future__ import annotations

import numpy as np


def uncertainty_shrunk_residual(
    predictions: np.ndarray,
    *,
    kind: str,
    power: float = 1.0,
    minimum_agreement: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(predictions, float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("predictions must have shape [models, rows]")
    mean = values.mean(axis=0)
    if kind == "confidence":
        spread = values.std(axis=0)
        gate = np.abs(mean) / (np.abs(mean) + spread + 1e-6)
        gate = np.power(np.clip(gate, 0.0, 1.0), float(power))
    elif kind == "sign_agreement":
        agreement = np.abs(np.sign(values).mean(axis=0))
        floor = float(minimum_agreement)
        gate = np.clip((agreement - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0)
    else:
        raise ValueError(f"Unknown uncertainty gate: {kind}")
    return mean * gate, gate

