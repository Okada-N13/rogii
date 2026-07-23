"""Build Stage 19C Colab packager and Kaggle submission notebooks."""

from __future__ import annotations

import json
from pathlib import Path


COLAB_OUTPUT = Path("notebooks/500_build_stage19c_inference_package.ipynb")
KAGGLE_SOURCE = Path("notebooks/470_kaggle_top_pf_a130_branch_safe.ipynb")
KAGGLE_OUTPUT = Path("notebooks/510_kaggle_top_pf_stage19_trajectory.ipynb")
INSERT_BEFORE = "# Final submission audit: verify the final file after all enabled correction layers."


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _source(cell: dict) -> str:
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def _markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(text)}


COLAB_CELLS = [
    _markdown(
        "# Stage 19C: Internet-OFF trajectory inference package\n\n"
        "Stage 19Bで合格した15個のHGBを、Kaggleのscikit-learn版に依存しないNumPy形式へ変換します。"
        "学習は行わず、変換前後の予測完全一致を検証してzip化します。CPUランタイムで実行してください。\n"
    ),
    _code("from google.colab import drive\ndrive.mount('/content/drive')\n"),
    _code(
        "from pathlib import Path\nimport json, os, shutil, subprocess\n"
        "REPOSITORY_URL='https://github.com/Okada-N13/rogii.git'\n"
        "repo_dir=Path('/content/ROGII'); drive_root=Path('/content/drive/MyDrive/kaggle/rogii')\n"
        "artifact_dir=drive_root/'artifacts'\n"
        "if not (repo_dir/'.git').is_dir(): subprocess.run(['git','clone',REPOSITORY_URL,str(repo_dir)],check=True)\n"
        "else:\n"
        "    subprocess.run(['git','-C',str(repo_dir),'pull','--ff-only','origin','main'],check=True)\n"
        "if shutil.which('uv') is None: subprocess.run(['bash','-lc','curl -LsSf https://astral.sh/uv/install.sh | sh'],check=True)\n"
        "os.environ['PATH']='/root/.local/bin:'+os.environ['PATH']\n"
        "subprocess.run(['uv','sync','--frozen'],cwd=repo_dir,check=True)\n"
        "def run_checked(command):\n"
        "    result=subprocess.run(command,cwd=repo_dir,text=True,capture_output=True)\n"
        "    if result.stdout: print(result.stdout,flush=True)\n"
        "    if result.returncode:\n"
        "        print(result.stderr,flush=True); raise RuntimeError(f'command failed: {command}')\n"
    ),
    _code(
        "stage19b_run=artifact_dir/'stage19b_trajectory_package_full_v001'\n"
        "assert json.loads((stage19b_run/'summary.json').read_text())['promoted_to_stage19c'] is True\n"
        "RUN_ID='stage19c_trajectory_inference_package_v001'; run_dir=artifact_dir/RUN_ID\n"
        "if not (run_dir/'summary.json').is_file():\n"
        "    run_checked(['uv','run','rogii-trajectory-inference-package','--stage19b-run',str(stage19b_run),"
        "'--artifact-dir',str(artifact_dir),'--run-id',RUN_ID])\n"
        "summary=json.loads((run_dir/'summary.json').read_text())\n"
        "{key:summary[key] for key in ['stage19c_package_complete','package_ready','portable_models',"
        "'portable_prediction_max_abs_difference','package_manifest_sha256','zip','gates','next_step']}\n"
    ),
    _markdown(
        "最後の辞書で `package_ready=True` と変換差 `0.0` を確認し、"
        "`stage19c_trajectory_inference_package.zip` をKaggle Dataset "
        "`rogii-stage19c-trajectory-inference-package` としてアップロードしてください。\n"
    ),
]


STAGE19_CELL = r'''# Stage 19C: target-safe three-coefficient trajectory residual.
import hashlib as _s19_hashlib
import importlib.util as _s19_importlib
import json as _s19_json
import zipfile as _s19_zipfile
from pathlib import Path as _S19Path
import pandas as _s19_pd

_S19_INPUT=_S19Path('/kaggle/input'); _S19_WORK=_S19Path('/kaggle/working')
_S19_SUBMISSION=_S19_WORK/'submission.csv'
def _s19_manifest(path):
    try: return bool(_s19_json.loads(path.read_text(encoding='utf-8')).get('stage19c_trajectory_inference_package'))
    except Exception: return False
_s19_manifests=[p for p in _S19_INPUT.rglob('manifest.json') if _s19_manifest(p)]
if not _s19_manifests:
    _s19_zips=sorted(_S19_INPUT.rglob('stage19c_trajectory_inference_package.zip'))
    if len(_s19_zips)!=1: raise AssertionError(f'Expected one Stage 19C package, found {_s19_zips}')
    _s19_extract=_S19_WORK/'stage19c_trajectory_inference_package'; _s19_extract.mkdir(parents=True,exist_ok=True)
    with _s19_zipfile.ZipFile(_s19_zips[0]) as archive: archive.extractall(_s19_extract)
    _s19_manifests=[p for p in _s19_extract.rglob('manifest.json') if _s19_manifest(p)]
if len(_s19_manifests)!=1: raise AssertionError(f'Expected one Stage 19C manifest, found {_s19_manifests}')
_s19_package=_s19_manifests[0].parent
_s19_manifest_data=_s19_json.loads(_s19_manifests[0].read_text(encoding='utf-8'))
if int(_s19_manifest_data.get('package_version',0))<2: raise AssertionError('Stage 19C package v2 is required')
_s19_module_path=_s19_package/_s19_manifest_data['inference_file']
if _s19_hashlib.sha256(_s19_module_path.read_bytes()).hexdigest()!=_s19_manifest_data['inference_sha256']:
    raise AssertionError('Stage 19C inference module hash mismatch')
_s19_samples=[p for p in _S19_INPUT.rglob('sample_submission.csv') if (p.parent/'train').is_dir() and (p.parent/'test').is_dir()]
if len(_s19_samples)!=1: raise AssertionError(f'Competition data not found uniquely: {_s19_samples}')
if not _S19_SUBMISSION.is_file(): raise AssertionError('Base submission.csv was not produced')
_s19_spec=_s19_importlib.spec_from_file_location('stage19_trajectory',_s19_module_path)
_s19_module=_s19_importlib.module_from_spec(_s19_spec); _s19_spec.loader.exec_module(_s19_module)
STAGE19C_TEST_AUDIT=_s19_module.apply_trajectory_residual(_s19_package,_s19_samples[0].parent,_S19_SUBMISSION)
_s19_expected=int(_s19_pd.read_csv(_s19_samples[0]).id.astype(str).str.rsplit('_',n=1).str[0].nunique())
if STAGE19C_TEST_AUDIT['wells']!=_s19_expected or STAGE19C_TEST_AUDIT['hidden_target_columns_used']:
    raise AssertionError(f'Stage 19C audit failed: {STAGE19C_TEST_AUDIT}')
print('STAGE19C_TEST_AUDIT =',STAGE19C_TEST_AUDIT,flush=True)
'''


def build() -> None:
    colab = {
        "cells": COLAB_CELLS,
        "metadata": {
            "accelerator": "CPU", "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "stage19c": {"submission": False, "standalone_setup": True, "portable_export": True},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    COLAB_OUTPUT.write_text(json.dumps(colab, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")

    notebook = json.loads(KAGGLE_SOURCE.read_text(encoding="utf-8"))
    positions = [i for i, cell in enumerate(notebook["cells"]) if INSERT_BEFORE in _source(cell)]
    if len(positions) != 1:
        raise RuntimeError(f"Expected one final audit marker, found {positions}")
    notebook["cells"].insert(positions[0], _code(STAGE19_CELL))
    notebook["cells"][0]["source"] = _lines(
        _source(notebook["cells"][0]) +
        "\n\nThis variant adds the OOF-promoted Stage 19 three-coefficient trajectory residual. "
        "It uses only test trajectory, GR, typewell and visible TVT_input, and requires the Stage 19C Dataset.\n"
    )
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"], cell["outputs"] = None, []
        source = _source(cell)
        if "'profile': 'top_pf_a130_branch_conservative_safe'" in source:
            cell["source"] = _lines(source.replace(
                "'profile': 'top_pf_a130_branch_conservative_safe'",
                "'profile': 'top_pf_a130_branch_conservative_stage19_trajectory'",
            ))
    notebook["metadata"]["stage19c_trajectory"] = {
        "base_public_lb": 6.589, "weight": .5, "cap_ft": 16., "internet": False,
        "hidden_target_columns_used": False,
        "required_dataset": "rogii-stage19c-trajectory-inference-package",
    }
    KAGGLE_OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")


if __name__ == "__main__":
    build()
