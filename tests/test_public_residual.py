from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.evaluation.gates import evaluate_candidate_gates
from rogii.models.public_residual import (
    apply_public_delta_postprocess,
    build_public_residual_features,
    crossfit_positive_ridge,
    crossfit_public_residual,
)


def _frame(n_wells: int = 10, rows_per_well: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    well = np.repeat([f"w{index:02d}" for index in range(n_wells)], rows_per_well)
    row = np.tile(np.arange(rows_per_well), n_wells)
    last = np.repeat(np.linspace(100.0, 110.0, n_wells), rows_per_well)
    target = 0.15 * row + np.repeat(np.linspace(-2.0, 2.0, n_wells), rows_per_well)
    return pd.DataFrame(
        {
            "id": [f"{name}_{index}" for name, index in zip(well, row, strict=True)],
            "well_id": well,
            "target": target,
            "last_known_tvt": last,
            "pf_ancc": last + 0.1 * row,
            "pf_ancc_std": rng.uniform(0.5, 2.0, len(well)),
            "md_since": row.astype(float) * 10.0,
            "frac": row / (rows_per_well - 1),
            "z": rng.normal(5000.0, 5.0, len(well)),
            "gr": rng.normal(80.0, 10.0, len(well)),
        }
    )


def test_public_ridge_and_residual_crossfit_are_aligned() -> None:
    frame = _frame()
    rng = np.random.default_rng(8)
    base_models = np.column_stack(
        [frame["target"].to_numpy() + rng.normal(0.0, scale, len(frame)) for scale in (1.0, 1.2, 1.4)]
    )
    ridge, folds = crossfit_positive_ridge(
        base_models,
        frame["target"].to_numpy(),
        frame["well_id"],
        n_splits=5,
        alpha=1.0,
        tol=1e-4,
    )
    frame["fold"] = folds
    base = apply_public_delta_postprocess(frame, ridge, alpha=1.0, tau_md=0.0, pf_weight=0.0)
    features = build_public_residual_features(
        frame,
        base_models,
        base,
        requested_columns=["pf_ancc", "pf_ancc_std", "md_since", "frac", "z", "gr"],
        max_rows_per_well=10,
    )
    prediction = crossfit_public_residual(
        features,
        folds,
        {
            "max_iter": 20,
            "max_leaf_nodes": 5,
            "min_samples_leaf": 5,
            "l2_regularization": 1.0,
        },
        [42],
        target_clip=20.0,
    )
    assert len(prediction) == len(frame)
    assert np.isfinite(prediction).all()
    for well_id, group in frame.groupby("well_id"):
        assert group["fold"].nunique() == 1, well_id


def test_candidate_gate_promotes_consistent_improvement() -> None:
    frame = _frame(n_wells=20, rows_per_well=10)
    rng = np.random.default_rng(12)
    true = frame["last_known_tvt"].to_numpy() + frame["target"].to_numpy()
    shared_error = rng.normal(0.0, 2.0, len(frame))
    folds = np.repeat(np.arange(5), 4 * 10)
    baseline = pd.DataFrame(
        {
            "id": frame["id"],
            "well_id": frame["well_id"],
            "MD": frame["md_since"],
            "y_true": true,
            "y_pred": true + shared_error,
            "fold": folds,
        }
    )
    candidate = baseline.copy()
    candidate["y_pred"] = true + 0.7 * shared_error
    result = evaluate_candidate_gates(
        baseline,
        candidate,
        minimum_standard_gain=0.05,
        bootstrap_resamples=200,
        seed=42,
    )
    assert result["promoted"] is True
    assert result["pooled_rmse_delta"] < 0.0
