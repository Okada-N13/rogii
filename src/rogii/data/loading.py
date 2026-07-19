from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd


HORIZONTAL_SUFFIX = "__horizontal_well.csv"


def well_id_from_path(path: str | Path) -> str:
    name = Path(path).name
    if not name.endswith(HORIZONTAL_SUFFIX):
        raise ValueError(f"Not a horizontal well CSV: {name}")
    return name[: -len(HORIZONTAL_SUFFIX)]


def discover_horizontal_wells(data_dir: str | Path, split: str) -> list[Path]:
    split_dir = Path(data_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing data split directory: {split_dir}")
    files = sorted(split_dir.glob(f"*{HORIZONTAL_SUFFIX}"))
    if not files:
        raise FileNotFoundError(f"No horizontal well CSV files in: {split_dir}")
    return files


def load_horizontal_well(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.insert(0, "row_index", range(len(frame)))
    frame.insert(0, "well_id", well_id_from_path(path))
    return frame


def iter_horizontal_wells(
    data_dir: str | Path,
    split: str,
    well_ids: Sequence[str] | None = None,
) -> Iterator[tuple[Path, pd.DataFrame]]:
    selected = set(well_ids) if well_ids is not None else None
    for path in discover_horizontal_wells(data_dir, split):
        if selected is None or well_id_from_path(path) in selected:
            yield path, load_horizontal_well(path)


def dataset_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()

