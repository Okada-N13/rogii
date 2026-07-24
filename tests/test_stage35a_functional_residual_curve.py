from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rogii.models.functional_residual import (
    curve_descriptor,
    equal_cut_optimal_alpha,
    resample_curve,
    restore_curve,
)


ROOT = Path(__file__).resolve().parents[1]


def test_curve_resampling_preserves_linear_shape() -> None:
    source = np.linspace(-3.0, 4.0, 17)
    grid = resample_curve(source, 96)
    restored = restore_curve(grid, len(source))
    assert np.allclose(restored, source, atol=1e-6)


def test_equal_cut_alpha_does_not_weight_long_curve_more() -> None:
    short_truth = np.ones(5)
    long_truth = -np.ones(50)
    bases = [np.zeros(5), np.zeros(50)]
    full = [2.0 * short_truth, 2.0 * long_truth]
    alpha = equal_cut_optimal_alpha(bases, full, [short_truth, long_truth], 0.75)
    assert np.isclose(alpha, 0.5)


def test_curve_descriptor_is_finite_and_stable() -> None:
    features = np.arange(30, dtype=float).reshape(6, 5) / 10.0
    candidates = {
        "top_pf_a130": np.linspace(100.0, 105.0, 6),
        "selector_a130": np.linspace(99.0, 106.0, 6),
    }
    vector, names = curve_descriptor(features, candidates, "top_pf_a130", 0.30)
    repeated, repeated_names = curve_descriptor(
        features, candidates, "top_pf_a130", 0.30
    )
    assert np.isfinite(vector).all()
    assert np.array_equal(vector, repeated)
    assert names == repeated_names


def test_stage35a_notebook_is_clean_cpu_only_and_reserved_safe() -> None:
    path = ROOT / "notebooks" / "720_run_stage35a_functional_residual_curve.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "rogii-functional-residual" in text
    assert "stage35a_functional_residual_curve.yaml" in text
    assert "stage29a_multicut_manifest_v001" in text
    assert "'fold_safe_basis'" in text
    assert "torch.cuda" not in text
    assert payload["metadata"]["stage35a"]["reserved_confirmation_used"] is False
    assert payload["metadata"]["stage35a"]["submission"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")
