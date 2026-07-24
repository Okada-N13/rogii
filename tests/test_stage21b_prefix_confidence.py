from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rogii.cli.prefix_confidence import calibration_penalties, confidence_choice


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_penalties_and_confidence_guard() -> None:
    rows = []
    for index in range(12):
        rows.extend(
            [
                {
                    "candidate": "top_pf_a130",
                    "inner_rmse": 2.0 + index * 0.01,
                    "outer_rmse": 7.0 + index * 0.01,
                },
                {
                    "candidate": "selector_a130",
                    "inner_rmse": 1.0 + index * 0.01,
                    "outer_rmse": 9.0 + index * 0.01,
                },
            ]
        )
    penalties, report = calibration_penalties(
        pd.DataFrame(rows), ["top_pf_a130", "selector_a130"], 0.5
    )
    assert len(report) == 2
    assert penalties["selector_a130"] > penalties["top_pf_a130"]
    selected, reason, corrected = confidence_choice(
        {"top_pf_a130": [2.0, 2.1], "selector_a130": [1.8, 1.9]},
        penalties,
        "top_pf_a130",
        0.5,
        0.0,
    )
    assert selected == "top_pf_a130"
    assert reason in {"base_ranked_first", "corrected_margin"}
    assert corrected["selector_a130"] > corrected["top_pf_a130"]


def test_confidence_gate_accepts_consistent_large_margin() -> None:
    selected, reason, _ = confidence_choice(
        {"top_pf_a130": [4.0, 4.2], "top_pf_a160": [2.0, 2.1]},
        {"top_pf_a130": 5.0, "top_pf_a160": 5.0},
        "top_pf_a130",
        0.5,
        0.0,
    )
    assert selected == "top_pf_a160"
    assert reason == "alternative_accepted"


def test_stage21b_notebook_is_clean_and_compiles() -> None:
    path = ROOT / "notebooks" / "550_run_stage21b_prefix_confidence.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "rogii-prefix-confidence" in text
    assert "stage21a_prefix_router_full_v001" in text
    assert text.count("'--exclude-run'") == 2
    assert payload["metadata"]["stage21b"]["submission"] is False
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell.get("source", [])), str(path), "exec")
