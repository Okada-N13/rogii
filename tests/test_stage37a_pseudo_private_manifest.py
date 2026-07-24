from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import yaml

from rogii.cli.pseudo_private_manifest import main


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "740_run_stage37a_pseudo_private_manifest.ipynb"


def _source(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def _write_fixture(root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    artifacts = root / "artifacts"
    stage16, stage17, split, design = (artifacts / name for name in ("stage16", "stage17", "split", "design"))
    for path in (stage16, stage17, split, design):
        path.mkdir(parents=True)
    training_wells = [f"t{index:03d}" for index in range(5)]
    design_wells = [f"d{index:03d}" for index in range(2)]
    confirmation_wells = [f"c{index:03d}" for index in range(3)]
    all_wells = training_wells + design_wells + confirmation_wells
    signature_columns = [
        "typewell_gr_mean", "typewell_gr_std", "typewell_gr_q10", "typewell_gr_q50",
        "typewell_gr_q90", "typewell_tvt_min", "typewell_tvt_max", "typewell_tvt_span",
    ]
    assignments = []
    for index, well_id in enumerate(all_wells):
        row = {
            "well_id": well_id, "fold": index % 5, "spatial_fold": index % 3,
            "branch_group_fold": index % 5, "branch_group": index,
        }
        row.update({column: float(index + offset + 1) for offset, column in enumerate(signature_columns)})
        assignments.append(row)
    pd.DataFrame(assignments).to_parquet(stage16 / "well_assignments.parquet", index=False)

    cuts = []
    training_seeds, design_seeds, confirmation_seeds = [], [], []
    for well_id in training_wells:
        for fraction in (0.2, 0.3):
            cut_id = f"{well_id}__{int(fraction * 100)}"
            cuts.append({
                "cut_id": cut_id, "well_id": well_id, "cut_index": int(fraction * 100),
                "requested_fraction": fraction, "suffix_rows": 100, "evaluation_role": "primary",
                "replay_eligible": True,
            })
            if fraction == 0.3:
                training_seeds.append({"cut_id": cut_id, "well_id": well_id})
    for well_id in design_wells:
        cut_id = f"{well_id}__30"
        cuts.append({
            "cut_id": cut_id, "well_id": well_id, "cut_index": 30, "requested_fraction": 0.3,
            "suffix_rows": 100, "evaluation_role": "primary", "replay_eligible": True,
        })
        design_seeds.append({"cut_id": cut_id, "well_id": well_id})
    for well_id in confirmation_wells:
        cut_id = f"{well_id}__30"
        cuts.append({
            "cut_id": cut_id, "well_id": well_id, "cut_index": 30, "requested_fraction": 0.3,
            "suffix_rows": 100, "evaluation_role": "primary", "replay_eligible": True,
        })
        confirmation_seeds.append({"cut_id": cut_id, "well_id": well_id})
    pd.DataFrame(cuts).to_parquet(stage17 / "cut_report.parquet", index=False)
    (stage17 / "summary.json").write_text(
        json.dumps({"stage16b_manifest_sha256": "synthetic"}), encoding="utf-8"
    )
    pd.DataFrame(training_seeds).to_parquet(split / "training_cut_ids.parquet", index=False)
    pd.DataFrame(confirmation_seeds).to_parquet(split / "confirmation_cut_ids.parquet", index=False)
    (split / "summary.json").write_text(
        json.dumps({"public_replay_eligible_only": True}), encoding="utf-8"
    )
    pd.DataFrame(design_seeds).to_parquet(design / "confidence_cut_report.parquet", index=False)
    config = {
        "seed": 42,
        "provenance": {"stage16b_manifest_sha256": "synthetic"},
        "primary_fractions": [0.2, 0.3],
        "validation": {"n_typewell_folds": 3},
        "expected": {"training_wells": 5, "design_wells": 2, "design_cuts": 2, "confirmation_wells": 3},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return artifacts, stage16, stage17, split, design, config_path


def test_stage37_manifest_freezes_disjoint_roles_without_target_reads(tmp_path: Path) -> None:
    artifacts, stage16, stage17, split, design, config = _write_fixture(tmp_path)
    arguments = [
        "--config", str(config), "--stage16b-run", str(stage16), "--stage17a-run", str(stage17),
        "--split-run", str(split), "--design-validation-run", str(design),
        "--artifact-dir", str(artifacts),
    ]
    main([*arguments, "--run-id", "first"])
    main([*arguments, "--run-id", "second"])
    first = json.loads((artifacts / "first" / "summary.json").read_text(encoding="utf-8"))
    second = json.loads((artifacts / "second" / "summary.json").read_text(encoding="utf-8"))
    assert first["promoted_to_stage37b_top_pf_replay"] is True
    assert first["pseudo_private_manifest_sha256"] == second["pseudo_private_manifest_sha256"]
    assert first["roles"]["training"]["cuts"] == 10
    assert first["roles"]["training"]["wells"] == 5
    assert first["roles"]["design_validation"]["wells"] == 2
    assert first["roles"]["confirmation_locked"]["wells"] == 3
    assert first["confirmation_target_columns_read"] is False
    assert all(first["gates"].values())
    manifest = pd.read_parquet(artifacts / "first" / "pseudo_private_manifest.parquet")
    role_wells = {
        role: set(frame["well_id"])
        for role, frame in manifest.groupby("benchmark_role", sort=True)
    }
    assert role_wells["training"].isdisjoint(role_wells["design_validation"])
    assert role_wells["training"].isdisjoint(role_wells["confirmation_locked"])


def test_stage37_notebook_is_clean_and_standalone() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join(_source(cell) for cell in payload["cells"])
    assert "rogii-pseudo-private-manifest" in text
    assert "stage37a_locked_split_manifest_v001" in text
    assert "rogii-scaled-emission-manifest" in text
    assert "'--training-wells-per-fold','100'" in text
    assert "'--confirmation-wells-per-fold','24'" in text
    assert "stage21b_prefix_confidence_full_v001" in text
    assert payload["metadata"]["accelerator"] == "CPU"
    assert payload["metadata"]["stage37a"]["confirmation_target_unread"] is True
    for index, cell in enumerate(payload["cells"]):
        assert cell.get("outputs", []) == []
        if cell.get("cell_type") == "code":
            ast.parse(_source(cell), filename=f"cell_{index}")
