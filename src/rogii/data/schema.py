from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


COMMON_REQUIRED_COLUMNS = {"MD", "X", "Y", "Z", "GR", "TVT_input"}
TRAIN_REQUIRED_COLUMNS = COMMON_REQUIRED_COLUMNS | {"TVT"}


class SchemaError(ValueError):
    """Raised when a competition file violates an expected invariant."""


@dataclass(frozen=True)
class WellStats:
    well_id: str
    split: str
    n_rows: int
    n_known: int
    n_target: int
    first_md: float
    last_md: float
    anchor_md: float
    first_target_md: float
    missing_gr: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def target_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["TVT_input"].isna()


def validate_horizontal_well(
    frame: pd.DataFrame,
    split: str,
    well_id: str | None = None,
) -> WellStats:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be train or test, got {split!r}")
    resolved_well_id = well_id or str(frame["well_id"].iloc[0])
    required = TRAIN_REQUIRED_COLUMNS if split == "train" else COMMON_REQUIRED_COLUMNS
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SchemaError(f"{resolved_well_id}: missing required columns {missing}")
    if frame.empty:
        raise SchemaError(f"{resolved_well_id}: file is empty")

    numeric_columns = sorted(required)
    non_numeric = [column for column in numeric_columns if not pd.api.types.is_numeric_dtype(frame[column])]
    if non_numeric:
        raise SchemaError(f"{resolved_well_id}: non-numeric columns {non_numeric}")
    if frame["MD"].isna().any() or frame["Z"].isna().any():
        raise SchemaError(f"{resolved_well_id}: MD and Z must not contain missing values")
    if not (frame["MD"].diff().dropna() > 0).all():
        raise SchemaError(f"{resolved_well_id}: MD must be strictly increasing")

    hidden = target_mask(frame).to_numpy()
    if not hidden.any() or hidden.all():
        raise SchemaError(f"{resolved_well_id}: TVT_input must contain a known prefix and hidden suffix")
    first_hidden = int(np.flatnonzero(hidden)[0])
    if (~hidden[first_hidden:]).any():
        raise SchemaError(f"{resolved_well_id}: known TVT_input appears after hidden suffix begins")

    if split == "train":
        if frame["TVT"].isna().any():
            raise SchemaError(f"{resolved_well_id}: training TVT contains missing values")
        known = ~hidden
        if not np.allclose(
            frame.loc[known, "TVT_input"].to_numpy(),
            frame.loc[known, "TVT"].to_numpy(),
            rtol=0.0,
            atol=1e-8,
        ):
            raise SchemaError(f"{resolved_well_id}: known TVT_input does not equal TVT")

    anchor_index = first_hidden - 1
    return WellStats(
        well_id=resolved_well_id,
        split=split,
        n_rows=len(frame),
        n_known=first_hidden,
        n_target=len(frame) - first_hidden,
        first_md=float(frame["MD"].iloc[0]),
        last_md=float(frame["MD"].iloc[-1]),
        anchor_md=float(frame["MD"].iloc[anchor_index]),
        first_target_md=float(frame["MD"].iloc[first_hidden]),
        missing_gr=int(frame["GR"].isna().sum()),
    )

