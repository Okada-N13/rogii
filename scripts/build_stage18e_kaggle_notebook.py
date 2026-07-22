"""Insert the Stage 18E ranked-retrieval postprocess into the frozen 6.685 V599 notebook."""

from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("notebooks/230_kaggle_v599_a130_frontier_safe.ipynb")
OUTPUT = Path("notebooks/460_kaggle_v599_stage18_ranked_retrieval.ipynb")
INSERT_BEFORE = "# Final submission audit: verify the final file after all enabled correction layers."


def _source(cell: dict) -> str:
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(value)


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


STAGE18_CELL = r'''# Stage 18E: fold-safe learned donor retrieval on top of the frozen V599 submission.
import hashlib as _s18_hashlib
import importlib.util as _s18_importlib
import json as _s18_json
import zipfile as _s18_zipfile
from pathlib import Path as _S18Path

_S18_INPUT = _S18Path('/kaggle/input')
_S18_WORK = _S18Path('/kaggle/working')
_S18_SUBMISSION = _S18_WORK / 'submission.csv'

def _s18_is_manifest(path):
    try:
        return bool(_s18_json.loads(path.read_text(encoding='utf-8')).get('stage18e_package'))
    except Exception:
        return False

_s18_manifests = [p for p in _S18_INPUT.rglob('manifest.json') if _s18_is_manifest(p)]
if not _s18_manifests:
    _s18_archives = sorted(_S18_INPUT.rglob('stage18e_ranked_retrieval_package.zip'))
    if len(_s18_archives) != 1:
        raise AssertionError(f'Expected one Stage 18E package manifest or zip, found zips={_s18_archives}')
    _s18_extract = _S18_WORK / 'stage18e_ranked_retrieval_package'
    _s18_extract.mkdir(parents=True, exist_ok=True)
    with _s18_zipfile.ZipFile(_s18_archives[0]) as _s18_bundle:
        _s18_bundle.extractall(_s18_extract)
    _s18_manifests = [p for p in _s18_extract.rglob('manifest.json') if _s18_is_manifest(p)]
if len(_s18_manifests) != 1:
    raise AssertionError(f'Expected one Stage 18E manifest, found {_s18_manifests}')
_s18_package = _s18_manifests[0].parent

_s18_samples = [
    p for p in _S18_INPUT.rglob('sample_submission.csv')
    if (p.parent / 'train').is_dir() and (p.parent / 'test').is_dir()
]
if len(_s18_samples) != 1:
    raise AssertionError(f'Competition data not found uniquely: {_s18_samples}')
_s18_data = _s18_samples[0].parent
if not _S18_SUBMISSION.is_file():
    raise AssertionError('Frozen V599 submission.csv was not produced before Stage 18E')
_s18_base_sha256 = _s18_hashlib.sha256(_S18_SUBMISSION.read_bytes()).hexdigest()

_s18_spec = _s18_importlib.spec_from_file_location('stage18_retrieval', _s18_package / 'stage18_retrieval.py')
if _s18_spec is None or _s18_spec.loader is None:
    raise ImportError('Could not load Stage 18E inference module')
_s18_module = _s18_importlib.module_from_spec(_s18_spec)
_s18_spec.loader.exec_module(_s18_module)
STAGE18E_TEST_AUDIT = _s18_module.apply_ranked_retrieval(_s18_package, _s18_data, _S18_SUBMISSION)
_s18_statuses = [row.get('status') for row in STAGE18E_TEST_AUDIT.get('well_report', [])]
if len(_s18_statuses) != 3 or any(status != 'applied' for status in _s18_statuses):
    raise AssertionError(f'Stage 18E was not applied to all 3 test wells: {_s18_statuses}')
STAGE18E_TEST_AUDIT['base_submission_sha256'] = _s18_base_sha256
(_S18_WORK / 'stage18_retrieval_audit.json').write_text(
    _s18_json.dumps(STAGE18E_TEST_AUDIT, indent=2), encoding='utf-8'
)
print('STAGE18E_TEST_AUDIT =', STAGE18E_TEST_AUDIT, flush=True)
'''


def build() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    positions = [index for index, cell in enumerate(notebook["cells"]) if INSERT_BEFORE in _source(cell)]
    if len(positions) != 1:
        raise RuntimeError(f"Expected one final-audit marker, found {positions}")
    title = _source(notebook["cells"][0]).replace(
        "ROGII V599 A130 branch-conservative — sanitized 6.768 frontier",
        "ROGII V599 + Stage 18 ranked retrieval — fold-safe submission build",
    )
    title += (
        "\n\nThis variant preserves the frozen 6.685 V599 pipeline and adds only the "
        "Stage 18D OOF-promoted, 20% fold-safe ranked donor retrieval after the branch hedge. "
        "It requires the Stage 18E package Dataset and runs with Internet OFF.\n"
    )
    notebook["cells"][0]["source"] = _lines(title)
    notebook["cells"].insert(positions[0], {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(STAGE18_CELL),
    })
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    for cell in notebook["cells"]:
        source = _source(cell)
        if "'profile': 'v599_a130_branch_conservative'" in source:
            cell["source"] = _lines(source.replace(
                "'profile': 'v599_a130_branch_conservative'",
                "'profile': 'v599_a130_branch_conservative_stage18_ranked_retrieval'",
            ))
    notebook["metadata"]["stage18e_ranked_retrieval"] = {
        "base_public_lb": 6.685, "blend_weight": 0.20, "selected_donors": 4,
        "same_well_target_transfer_removed": True, "internet": False,
        "required_dataset": "rogii-stage18e-ranked-retrieval-package",
        "package_manifest_sha256": "7bddc1914f3d046b678dbb8f5d1cc17427b03bc85c1a06d1f2088cbe68d3935d",
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
