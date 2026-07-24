from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rogii.models.residual_cut_gate import cut_gate_features, optimal_gate


ROOT = Path(__file__).resolve().parents[1]


def test_optimal_gate_recovers_interpolation_coefficient() -> None:
    truth = np.array([1.0, 2.0, 3.0])
    base = np.zeros(3)
    full = 2.0 * truth
    assert np.isclose(optimal_gate(base, full, truth), 0.5)
    assert optimal_gate(base, base, truth) == 0.0


def test_cut_gate_features_are_finite_and_schema_stable() -> None:
    normalized = np.arange(24, dtype=float).reshape(6, 4) / 10.0
    residual = np.linspace(-2.0, 3.0, 6)
    candidates = {
        "top_pf_a130": np.linspace(100.0, 105.0, 6),
        "selector_a130": np.linspace(99.0, 106.0, 6),
        "public_oof": np.linspace(101.0, 104.0, 6),
    }
    vector, names = cut_gate_features(
        normalized, residual, candidates, "top_pf_a130", 0.30
    )
    assert len(vector) == len(names)
    assert np.isfinite(vector).all()
    assert names[:2] == ["requested_fraction", "log_suffix_rows"]
    changed, changed_names = cut_gate_features(
        normalized, residual, candidates, "top_pf_a130", 0.30
    )
    assert np.array_equal(vector, changed)
    assert names == changed_names


def test_stage33a_notebook_is_clean_and_reserved_safe() -> None:
    path = ROOT / "notebooks" / "700_run_stage33a_tcn_cut_benefit_gate.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "'-m', 'rogii.cli.residual_cut_benefit_gate'" in text
    assert "stage33a_tcn_cut_benefit_gate.yaml" in text
    assert "stage29a_multicut_manifest_v001" in text
    assert "stage30a_residual_tcn_full_v001" in text
    assert payload["metadata"]["stage33a"]["reserved_confirmation_used"] is False
    assert payload["metadata"]["stage33a"]["submission"] is False
    assert payload["metadata"]["stage33a"]["tcn_training"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")
