from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return config


def resolve_data_dir(config: dict[str, Any] | None = None) -> Path:
    configured = (config or {}).get("data_dir")
    return Path(configured or os.environ.get("ROGII_DATA_DIR", "data")).expanduser().resolve()


def resolve_artifact_dir(config: dict[str, Any] | None = None) -> Path:
    configured = (config or {}).get("artifact_dir")
    return Path(
        configured or os.environ.get("ROGII_ARTIFACT_DIR", "artifacts")
    ).expanduser().resolve()

