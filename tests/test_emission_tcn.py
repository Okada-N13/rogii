from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")


def test_candidate_emission_tcn_scores_every_state_and_backpropagates() -> None:
    from rogii.models.emission_tcn import CandidateEmissionTCN, _loss

    offsets = np.arange(-60, 61, 2, dtype=np.float32)
    model = CandidateEmissionTCN(4, 4, offsets, {"channels": 8, "blocks": 2})
    costs = torch.rand(2, 11, 4, 61)
    rows = torch.rand(2, 11, 4)
    target = torch.randint(0, 61, (2, 11))
    logits = model(costs, rows)
    loss = _loss(logits, costs, target, {})
    loss.backward()
    assert logits.shape == (2, 11, 61)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
