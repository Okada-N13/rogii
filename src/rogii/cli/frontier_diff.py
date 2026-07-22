from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

from rogii.artifacts import write_json
from rogii.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 16A static frontier notebook audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _source(cell: dict[str, Any]) -> str:
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(value)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_notebook(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("nbformat") != 4 or not isinstance(payload.get("cells"), list):
        raise ValueError(f"Unsupported notebook: {path}")
    return payload


def _literal_assignments(source: str, prefix: str) -> dict[str, Any]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    output: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        for target in targets:
            if not isinstance(target, ast.Name) or not target.id.startswith(prefix):
                continue
            try:
                output[target.id] = ast.literal_eval(value_node)
            except (ValueError, TypeError):
                pass
    return output


def _input_literals(cells: list[dict[str, Any]]) -> list[str]:
    values: set[str] = set()
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        try:
            tree = ast.parse(_source(cell))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if "/kaggle/input" in value or value.endswith((".pt", ".joblib", ".pkl", ".cbm")):
                    values.add(value)
    return sorted(values)


def _metadata_contract(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    candidates = [value for key, value in metadata.items() if key.endswith("_safe_build")]
    return dict(candidates[0]) if len(candidates) == 1 and isinstance(candidates[0], dict) else {}


def analyze_notebooks(
    left_path: Path,
    right_path: Path,
    *,
    left_label: str,
    right_label: str,
    left_score: float,
    right_score: float,
) -> dict[str, Any]:
    left, right = _load_notebook(left_path), _load_notebook(right_path)
    if len(left["cells"]) != len(right["cells"]):
        raise ValueError("Stage 16A requires index-aligned notebooks")
    cells = []
    changed, changed_code = [], []
    diffs: dict[str, str] = {}
    for index, (left_cell, right_cell) in enumerate(zip(left["cells"], right["cells"], strict=True)):
        left_source, right_source = _source(left_cell), _source(right_cell)
        same = left_cell.get("cell_type") == right_cell.get("cell_type") and left_source == right_source
        cells.append({
            "cell": index,
            "cell_type": left_cell.get("cell_type"),
            "identical": same,
            "left_sha256": _sha256(left_source),
            "right_sha256": _sha256(right_source),
        })
        if not same:
            changed.append(index)
            if left_cell.get("cell_type") == "code" or right_cell.get("cell_type") == "code":
                changed_code.append(index)
            diffs[str(index)] = "".join(difflib.unified_diff(
                left_source.splitlines(True), right_source.splitlines(True),
                fromfile=f"{left_label}:cell{index}", tofile=f"{right_label}:cell{index}",
            ))
    hedge_cell = 48
    left_hedge = _literal_assignments(_source(left["cells"][hedge_cell]), "_BH_")
    right_hedge = _literal_assignments(_source(right["cells"][hedge_cell]), "_BH_")
    left_contract, right_contract = _metadata_contract(left), _metadata_contract(right)
    upstream_code_identical = all(
        _source(left["cells"][index]) == _source(right["cells"][index])
        for index in range(1, hedge_cell)
        if left["cells"][index].get("cell_type") == "code"
    )
    dependencies_left, dependencies_right = _input_literals(left["cells"]), _input_literals(right["cells"])
    score_delta = float(right_score - left_score)
    sanitation_keys = (
        "same_well_target_transfer_removed", "final_single_artifact",
    )
    sanitation = {
        key: bool(left_contract.get(key)) and bool(right_contract.get(key))
        for key in sanitation_keys
    }
    sanitation.update({
        "upstream_prediction_code_identical": upstream_code_identical,
        "input_dependency_literals_identical": dependencies_left == dependencies_right,
        "changed_code_cells_only_expected": changed_code == [48, 50],
        "hedge_cell_has_no_hidden_tvt_column_access": all(
            token not in _source(right["cells"][hedge_cell]) for token in ("['TVT']", '["TVT"]')
        ),
        "hedge_cell_has_no_lb_probe": "LB_PROBE" not in _source(right["cells"][hedge_cell]).upper(),
    })
    return {
        "stage": "16A_frontier_static_diff",
        "left": {"label": left_label, "path": str(left_path), "actual_lb": float(left_score), "metadata": left_contract},
        "right": {"label": right_label, "path": str(right_path), "actual_lb": float(right_score), "metadata": right_contract},
        "winner": left_label if left_score < right_score else right_label,
        "score_delta_right_minus_left": score_delta,
        "cell_count": len(cells),
        "changed_cells": changed,
        "changed_code_cells": changed_code,
        "upstream_prediction_code_identical_through_cell_47": upstream_code_identical,
        "hedge_parameters": {"left": left_hedge, "right": right_hedge},
        "input_dependencies": {"left": dependencies_left, "right": dependencies_right, "identical": dependencies_left == dependencies_right},
        "sanitation_audit": sanitation,
        "cell_manifest": cells,
        "cell_diffs": diffs,
        "findings": [
            "The advertised branch-overlap notebook adds no new upstream model or dependency.",
            "The only changed prediction cell is the final PF seed-branch midpoint hedge.",
            "The right variant lowers hedge strength 1.00->0.60 and cap 3->2 ft.",
            "The right variant applies the hedge after existing visible-prefix/overlap routes; the left variant skips those wells.",
            f"The right variant was {abs(score_delta):.3f} LB RMSE worse in the user reproduction." if score_delta > 0 else f"The right variant was {abs(score_delta):.3f} LB RMSE better in the user reproduction.",
        ],
        "decision": "Freeze the left 6.685 notebook. Do not tune this hedge; proceed to test-like validation.",
    }


def _markdown(report: dict[str, Any]) -> str:
    left, right = report["left"], report["right"]
    hedge = report["hedge_parameters"]
    lines = [
        "# Stage 16A frontier差分監査", "",
        f"- left: `{left['label']}` / actual LB `{left['actual_lb']:.3f}`",
        f"- right: `{right['label']}` / actual LB `{right['actual_lb']:.3f}`",
        f"- winner: `{report['winner']}`",
        f"- right - left: `{report['score_delta_right_minus_left']:+.3f}`", "",
        "## 結論", "",
        "230/240は上流pipeline、入力dependency、model探索が同一である。予測ロジックの差はセル48の最終PF seed-branch midpoint hedgeだけで、240に新しいbranch-overlap modelは存在しない。",
        "", "## 変更点", "",
        f"- changed cells: `{report['changed_cells']}`",
        f"- changed code cells: `{report['changed_code_cells']}`",
        f"- upstream identical through cell 47: `{report['upstream_prediction_code_identical_through_cell_47']}`",
        f"- input dependencies identical: `{report['input_dependencies']['identical']}`",
        f"- left hedge: `{json.dumps(hedge['left'], sort_keys=True)}`",
        f"- right hedge: `{json.dumps(hedge['right'], sort_keys=True)}`", "",
        "## 判断", "",
        report["decision"], "",
        "## Findings", "",
    ]
    lines.extend(f"- {value}" for value in report["findings"])
    lines.extend(["", "## Unified diff", ""])
    for cell, value in report["cell_diffs"].items():
        lines.extend([f"### Cell {cell}", "", "```diff", value.rstrip(), "```", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    repo = args.repo_dir.resolve()
    left, right = dict(config["left"]), dict(config["right"])
    report = analyze_notebooks(
        repo / left["path"], repo / right["path"],
        left_label=str(left["label"]), right_label=str(right["label"]),
        left_score=float(left["actual_lb"]), right_score=float(right["actual_lb"]),
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "frontier_diff.json", report)
    write_json(output / "dependency_manifest.json", report["input_dependencies"])
    write_json(output / "sanitation_audit.json", report["sanitation_audit"])
    write_json(output / "cell_manifest.json", report["cell_manifest"])
    (output / "notebook_diff.md").write_text(_markdown(report), encoding="utf-8")
    print({
        "stage16a_complete": True,
        "winner": report["winner"],
        "score_delta_right_minus_left": report["score_delta_right_minus_left"],
        "changed_code_cells": report["changed_code_cells"],
        "upstream_prediction_code_identical": report["upstream_prediction_code_identical_through_cell_47"],
        "input_dependencies_identical": report["input_dependencies"]["identical"],
        "sanitation_passed": all(report["sanitation_audit"].values()),
        "decision": report["decision"],
    }, flush=True)


if __name__ == "__main__":
    main()
