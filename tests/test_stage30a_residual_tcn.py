from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_residual_tcn_preserves_sequence_shape() -> None:
    torch = pytest.importorskip("torch")
    from rogii.models.residual_tcn import ResidualTCN

    model = ResidualTCN(27, channels=8, blocks=2, kernel_size=3, dropout=0.0)
    output = model(torch.zeros(3, 41, 27))
    assert tuple(output.shape) == (3, 41)
    assert torch.isfinite(output).all()


def test_stage30a_notebook_is_clean_and_gpu_safe() -> None:
    path = ROOT / "notebooks" / "670_run_stage30a_residual_tcn.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "'-m','rogii.cli.sequence_residual_field'" in text
    assert "stage30a_residual_tcn.yaml" in text
    assert "PYTHONPATH" in text
    assert "torch.cuda.is_available()" in text
    assert payload["metadata"]["stage30a"]["reserved_confirmation_used"] is False
    assert payload["metadata"]["stage30a"]["submission"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")

