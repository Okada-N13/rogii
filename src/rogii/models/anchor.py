from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.data.schema import target_mask


def predict_anchor(frame: pd.DataFrame, mode: str = "last_tvt_anchor") -> pd.DataFrame:
    """Predict a hidden suffix from the last visible TVT using a simple anchor rule."""
    hidden = target_mask(frame).to_numpy()
    if not hidden.any() or hidden.all():
        raise ValueError("TVT_input must contain a known prefix and hidden suffix")
    first_hidden = int(np.flatnonzero(hidden)[0])
    if (~hidden[first_hidden:]).any():
        raise ValueError("TVT_input mask must be a contiguous hidden suffix")

    anchor = frame.iloc[first_hidden - 1]
    anchor_tvt = float(anchor["TVT_input"])
    anchor_surface = float(anchor_tvt + anchor["Z"])
    target = frame.loc[hidden].copy()
    if mode == "last_tvt_anchor":
        prediction = np.full(len(target), anchor_tvt, dtype=float)
        anchor_value = anchor_tvt
    elif mode == "flat_surface_anchor":
        prediction = anchor_surface - target["Z"].to_numpy(dtype=float)
        anchor_value = anchor_surface
    else:
        raise ValueError(f"Unknown anchor mode: {mode!r}")

    result = pd.DataFrame(
        {
            "id": target["well_id"].astype(str) + "_" + target["row_index"].astype(str),
            "well_id": target["well_id"].astype(str).to_numpy(),
            "row_index": target["row_index"].to_numpy(dtype=np.int64),
            "MD": target["MD"].to_numpy(dtype=float),
            "Z": target["Z"].to_numpy(dtype=float),
            "model": mode,
            "anchor_value": anchor_value,
            "surface_anchor": anchor_surface,
            "y_pred": prediction,
        }
    )
    if "TVT" in target:
        result["y_true"] = target["TVT"].to_numpy(dtype=float)
    return result


def predict_last_tvt_anchor(frame: pd.DataFrame) -> pd.DataFrame:
    """Hold the last visible TVT constant over the hidden suffix."""
    return predict_anchor(frame, mode="last_tvt_anchor")


def predict_flat_surface_anchor(frame: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic: hold TVT+Z constant over the hidden suffix."""
    return predict_anchor(frame, mode="flat_surface_anchor")
