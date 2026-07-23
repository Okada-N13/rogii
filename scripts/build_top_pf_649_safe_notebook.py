"""Build the target-safe reproduction of the reported 6.49 top-PF public notebook."""

from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("notebooks/230_kaggle_v599_a130_frontier_safe.ipynb")
BRANCH_SOURCE = Path("notebooks/240_kaggle_branch_overlap_6594_safe.ipynb")
OUTPUT = Path("notebooks/470_kaggle_top_pf_a130_branch_safe.ipynb")


def _source(cell: dict) -> str:
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(value)


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one {label} marker, found {text.count(old)}")
    return text.replace(old, new)


def build() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    title = _source(notebook["cells"][0]).replace(
        "ROGII V599 A130 branch-conservative — sanitized 6.768 frontier",
        "ROGII top-PF A130 branch-conservative — target-safe 6.49 reproduction",
    )
    title += (
        "\n\nThis is the target-safe reproduction of the public notebook reported at 6.49. "
        "Relative to the measured 6.685 sanitized V599 control, it changes only the hidden-test-active "
        "PF GR likelihood scale and conservative branch hedge. Same-well train TVT transfer, contact "
        "override, overlap probes, leaderboard probes, and read-only full-stack diagnostics remain absent.\n"
    )
    notebook["cells"][0]["source"] = _lines(title)

    pf_marker = "gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10., 60.))"
    pf_positions = [i for i, cell in enumerate(notebook["cells"]) if pf_marker in _source(cell)]
    if pf_positions != [36]:
        raise RuntimeError(f"Unexpected learned-PF cell positions: {pf_positions}")
    pf_source = _source(notebook["cells"][36])
    pf_source = _replace_once(
        pf_source,
        pf_marker,
        f"{pf_marker} * 1.3",
        "PF GR scale",
    )
    notebook["cells"][36]["source"] = _lines(pf_source)

    branch_notebook = json.loads(BRANCH_SOURCE.read_text(encoding="utf-8"))
    branch_source = _source(branch_notebook["cells"][48])
    for marker in ("_BH_STRENGTH = 0.60", "_BH_CAP = 2.00", "_BH_SKIP_EXISTING = False"):
        if marker not in branch_source:
            raise RuntimeError(f"Missing sanitized branch marker: {marker}")
    notebook["cells"][48]["source"] = _lines(branch_source)

    for cell in notebook["cells"]:
        source = _source(cell)
        if "'profile': 'v599_a130_branch_conservative'" in source:
            cell["source"] = _lines(source.replace(
                "'profile': 'v599_a130_branch_conservative'",
                "'profile': 'top_pf_a130_branch_conservative_safe'",
            ))
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    notebook["metadata"]["top_pf_649_safe_build"] = {
        "reported_public_score": 6.49,
        "measured_parent_score": 6.685,
        "sanitized_parent": SOURCE.name,
        "reported_source": "top-pf-config-branch-conservative(1).ipynb",
        "pf_gr_likelihood_scale_multiplier": 1.3,
        "branch_strength": 0.60,
        "branch_cap": 2.0,
        "branch_skip_existing": False,
        "same_well_target_transfer_removed": True,
        "contact_override_removed": True,
        "source_only_diagnostics_not_imported": True,
        "internet": False,
        "stage18_included": False,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
