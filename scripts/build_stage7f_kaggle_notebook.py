"""Build the Internet-off Stage 7F Kaggle notebook.

The reference JSON is the Kaggle API response for
mitubant/r005-mha120sep4mpkg10-exact-repro.  Its model-package inference
implementation is retained, while its LB-tuned final-output gate is replaced
by the Stage 7E OOF-promoted branch-local blend.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SOURCE = Path("notebooks/95_kaggle_public_mha_safe.ipynb")
OUTPUT = Path("notebooks/150_kaggle_public_robust_blend.ipynb")
REFERENCE_MARKER = "def _mp_build_submission()"
EXECUTION_MARKER = "    _mp_pkg_sub, _mp_pred_report"


def _source(cell: dict) -> str:
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(value)


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def build(reference_json: Path) -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    response = json.loads(reference_json.read_text(encoding="utf-8"))
    reference = json.loads(response["blob"]["source"])
    candidates = [c for c in reference["cells"] if REFERENCE_MARKER in _source(c)]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one model-package cell, found {len(candidates)}")

    code = _source(candidates[0])
    code = code[: code.index(EXECUTION_MARKER)]
    code = re.sub(
        r"\A# === MPKG-GATE PORT.*?(?=RUN_MODEL_PACKAGE_CORRECTION)",
        "# Stage 7F: exact pilkwang model-package inference, OOF-promoted branch-local blend.\n"
        "# Source lineage: mitubant/r005-mha120sep4mpkg10-exact-repro.\n",
        code,
        count=1,
        flags=re.DOTALL,
    )
    code = re.sub(
        r"MODEL_PACKAGE_GATED_MAX_WEIGHT = .*?\nMODEL_PACKAGE_GATED_SCALE = .*?\n"
        r"MODEL_PACKAGE_GATED_CANDIDATES = .*?\n",
        "MODEL_PACKAGE_BRANCH_WEIGHT = 0.4\n",
        code,
        count=1,
    )
    old_base = """    if not _mp_final_output.exists():
        raise RuntimeError(f'Base submission for model package correction was not produced: {_mp_final_output}')
    _mp_sample = _mp_pd.read_csv(_mp_sample_path)[['id']]
    _mp_base = _mp_sample.merge(_mp_pd.read_csv(_mp_final_output)[['id', 'tvt']], on='id', how='left')
    if _mp_base['tvt'].isna().any():
        raise RuntimeError('Base submission has missing sample ids before model package correction.')
    _mp_base.to_csv(_mp_work / 'submission_before_model_package.csv', index=False)
"""
    new_base = """    _mp_sample = _mp_pd.read_csv(_mp_sample_path)[['id']]
    _mp_base = _mp_sample.merge(sub_1[['id', 'tvt']], on='id', how='left')
    if _mp_base['tvt'].isna().any():
        raise RuntimeError('ravaghi branch has missing sample ids before model-package blend.')
"""
    if old_base not in code:
        raise RuntimeError("Reference base-submission block changed")
    code = code.replace(old_base, new_base, 1)
    code += """    _mp_pkg_sub, _mp_pred_report, _mp_weight_report, _mp_info = _mp_build_submission()
    if _mp_pkg_sub is None:
        raise RuntimeError('The pilkwang model-package dataset is required for Stage 7F.')

    _mp_joined = _mp_base.rename(columns={'tvt': 'tvt_ravaghi'}).merge(
        _mp_pkg_sub.rename(columns={'tvt': 'tvt_package'}), on='id', how='inner'
    )
    if len(_mp_joined) != len(_mp_sample):
        raise RuntimeError('Model-package/ravaghi id mismatch.')
    _mp_weight = float(MODEL_PACKAGE_BRANCH_WEIGHT)
    _mp_joined['tvt'] = (
        (1.0 - _mp_weight) * _mp_joined['tvt_ravaghi'].to_numpy(dtype=float)
        + _mp_weight * _mp_joined['tvt_package'].to_numpy(dtype=float)
    )
    sub_1 = _mp_validate_submission_ids(_mp_joined[['id', 'tvt']], _mp_sample, 'stage7f_ravaghi_branch')
    _mp_diff = _mp_joined['tvt_package'].to_numpy(dtype=float) - _mp_joined['tvt_ravaghi'].to_numpy(dtype=float)
    STAGE7F_PUBLIC_BLEND_AUDIT = {
        'oof_promoted': True,
        'branch': 'package_postprocessed',
        'branch_weight': _mp_weight,
        'base_branch': 'ravaghi',
        'rows': int(len(sub_1)),
        'package_difference_rmse': float(_mp_np.sqrt(_mp_np.mean(_mp_diff ** 2))),
        'effective_weight_before_mha': float(_mp_weight * 0.3 * 0.55),
        **_mp_info,
    }
    (_mp_work / 'stage7f_public_blend_audit.json').write_text(
        json.dumps(STAGE7F_PUBLIC_BLEND_AUDIT, indent=2, default=str)
    )
    display(_mp_weight_report)
    display(_mp_pred_report)
    print(STAGE7F_PUBLIC_BLEND_AUDIT)
"""

    intro = _source(notebook["cells"][0]).replace(
        "ROGII public MHA — safe submission build",
        "ROGII Stage 7F — OOF-promoted public model-package blend",
    )
    intro += (
        "\n\nThis variant replaces only the ravaghi branch with the Stage 7E "
        "OOF-promoted 60/40 blend. It requires the pilkwang "
        "`rogii-model-package` Kaggle dataset and runs with Internet OFF.\n"
    )
    notebook["cells"][0]["source"] = _lines(intro)
    insert_at = next(
        i + 1 for i, cell in enumerate(notebook["cells"])
        if "sub_1 = (sample_sub" in _source(cell)
    )
    package_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(code),
    }
    notebook["cells"].insert(insert_at, package_cell)
    notebook["metadata"]["stage7f_public_blend"] = {
        "oof_gate": "stage7e_public_robust_blend",
        "branch": "package_postprocessed",
        "branch_weight": 0.4,
        "base_branch": "ravaghi",
        "effective_weight_before_mha": 0.066,
        "required_dataset": "pilkwang/rogii-model-package",
        "internet": False,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_json", type=Path)
    args = parser.parse_args()
    build(args.reference_json)
