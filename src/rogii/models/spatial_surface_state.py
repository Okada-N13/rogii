from __future__ import annotations

import numpy as np
import pandas as pd


def anchored_spatial_plane(
    horizontal: pd.DataFrame,
    cut_index: int,
    *,
    window_ft: float = 1200.0,
    ridge: float = 0.05,
    robust_iterations: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Continue visible ``U=TVT+Z`` as an anchor-preserving XY plane.

    Complete XYZ geometry is allowed at inference, but TVT is read only before
    ``cut_index``.  The returned gradient is expressed as dU/dX and dU/dY.
    """
    cut = int(cut_index)
    if cut < 4 or cut >= len(horizontal):
        raise ValueError("cut_index must leave at least four prefix rows and one suffix row")
    prefix = horizontal.iloc[:cut]
    suffix = horizontal.iloc[cut:]
    md = prefix["MD"].to_numpy(float)
    use = md >= md[-1] - float(window_ft)
    if int(use.sum()) < 4:
        use = np.ones(len(prefix), dtype=bool)
    anchor = prefix.iloc[-1]
    xy = prefix.loc[use, ["X", "Y"]].to_numpy(float)
    design = xy - np.array([float(anchor["X"]), float(anchor["Y"])])
    u = (
        prefix.loc[use, "TVT"].to_numpy(float)
        + prefix.loc[use, "Z"].to_numpy(float)
        - float(anchor["TVT"] + anchor["Z"])
    )
    scale = max(float(np.sqrt(np.mean(np.square(design)))), 1.0)
    penalty = np.eye(2) * float(ridge) * scale * scale
    weights = np.ones(len(u), dtype=float)
    gradient = np.zeros(2, dtype=float)
    for _ in range(max(1, int(robust_iterations))):
        weighted = design * weights[:, None]
        gradient = np.linalg.solve(
            design.T @ weighted + penalty,
            design.T @ (weights * u),
        )
        residual = u - design @ gradient
        robust_scale = max(
            1.4826 * float(np.median(np.abs(residual - np.median(residual)))),
            1e-6,
        )
        weights = 1.0 / (1.0 + np.square(residual / (2.5 * robust_scale)))
    suffix_xy = suffix[["X", "Y"]].to_numpy(float)
    delta_u = (suffix_xy - np.array([float(anchor["X"]), float(anchor["Y"])])) @ gradient
    prediction = float(anchor["TVT"] + anchor["Z"]) + delta_u - suffix["Z"].to_numpy(float)
    if not np.isfinite(prediction).all() or not np.isfinite(gradient).all():
        raise RuntimeError("Spatial plane continuation produced non-finite values")
    return prediction, gradient


def guarded_spatial_blend(
    base: np.ndarray,
    plane: np.ndarray,
    *,
    weight: float,
    cap_ft: float,
    ramp_rows: float,
) -> np.ndarray:
    base = np.asarray(base, float)
    plane = np.asarray(plane, float)
    if base.shape != plane.shape:
        raise ValueError("base and plane predictions must have identical shape")
    step = np.arange(1, len(base) + 1, dtype=float)
    ramp = 1.0 - np.exp(-step / max(float(ramp_rows), 1.0))
    move = np.clip(float(weight) * (plane - base), -float(cap_ft), float(cap_ft))
    return base + ramp * move

