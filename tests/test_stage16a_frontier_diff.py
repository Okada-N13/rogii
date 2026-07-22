from __future__ import annotations

from pathlib import Path

from rogii.cli.frontier_diff import analyze_notebooks


ROOT = Path(__file__).resolve().parents[1]


def test_real_frontier_diff_is_only_final_hedge() -> None:
    report = analyze_notebooks(
        ROOT / "notebooks" / "230_kaggle_v599_a130_frontier_safe.ipynb",
        ROOT / "notebooks" / "240_kaggle_branch_overlap_6594_safe.ipynb",
        left_label="v599", right_label="advertised_6594",
        left_score=6.685, right_score=6.693,
    )
    assert report["winner"] == "v599"
    assert report["changed_cells"] == [0, 48, 50]
    assert report["changed_code_cells"] == [48, 50]
    assert report["upstream_prediction_code_identical_through_cell_47"] is True
    assert report["input_dependencies"]["identical"] is True
    assert report["hedge_parameters"]["left"]["_BH_STRENGTH"] == 1.0
    assert report["hedge_parameters"]["right"]["_BH_STRENGTH"] == 0.6
    assert all(report["sanitation_audit"].values())
