from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rogii.models.affine_path_state import (
    affine_path_library,
    aggregate_path_costs,
    decode_path_scores,
)


ROOT = Path(__file__).resolve().parents[1]


def test_affine_path_library_has_expected_endpoints() -> None:
    paths, specifications = affine_path_library(5, np.array([-2.0, 0.0, 2.0]))
    assert paths.shape == (5, 9)
    assert specifications.shape == (9, 2)
    assert np.allclose(paths[0], specifications[:, 0])
    assert np.allclose(paths[-1], specifications[:, 1])


def test_path_costs_and_soft_decoder_follow_low_cost_path() -> None:
    offsets = np.array([-1.0, 0.0, 1.0])
    paths, _ = affine_path_library(4, offsets)
    row_costs = np.tile(np.array([[2.0, 0.0, 2.0]]), (4, 1))
    scores, coverage = aggregate_path_costs(row_costs, offsets, paths)
    decoded = decode_path_scores(scores, paths, "soft", temperature=0.05)
    assert np.isfinite(scores).all() and np.isfinite(decoded).all()
    assert np.all(coverage == 1.0)
    assert np.max(np.abs(decoded)) < 0.1


def test_stage26a_notebook_is_clean_and_cpu_only() -> None:
    path = ROOT / "notebooks" / "630_run_stage26a_affine_path_state.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "rogii-affine-path-state" in text
    assert "stage26a_affine_path_state.yaml" in text
    assert payload["metadata"]["stage26a"]["submission"] is False
    assert payload["metadata"]["stage26a"]["training"] is False
    assert payload["metadata"]["stage26a"]["reserved_confirmation_used"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")
