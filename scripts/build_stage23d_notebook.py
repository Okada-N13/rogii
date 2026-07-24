"""Build the standalone Stage 23D hierarchical decoder notebook."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("notebooks/600_run_stage23d_hierarchical_decoder.ipynb")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build() -> None:
    cells = [
        markdown(
            "# Stage 23D: hierarchical emission decoder\n\n"
            "Stage 23BのrankerとStage 23Cで保存したOOF posteriorを再利用し、offsetを直接回帰せず、"
            "移動有無・方向・絶対量に分けて校正します。Stage 23Cのvolumeを再計算せず、GPUも不要です。"
            "Stage 21BはすでにStage 23B/Cの設計判断に使われているため、ここではdesign validation"
            "として扱います。通過してもKaggle提出せず、新しいwell集合で確認します。\n"
        ),
        code("from google.colab import drive\ndrive.mount('/content/drive')\n"),
        code(
            "from pathlib import Path\nimport json,os,shutil,subprocess\n"
            "REPOSITORY_URL='https://github.com/Okada-N13/rogii.git'\n"
            "repo_dir=Path('/content/ROGII'); drive_root=Path('/content/drive/MyDrive/kaggle/rogii')\n"
            "artifact_dir=drive_root/'artifacts'\n"
            "if not (repo_dir/'.git').is_dir(): subprocess.run(['git','clone',REPOSITORY_URL,str(repo_dir)],check=True)\n"
            "else: subprocess.run(['git','-C',str(repo_dir),'pull','--ff-only','origin','main'],check=True)\n"
            "if shutil.which('uv') is None: subprocess.run(['bash','-lc','curl -LsSf https://astral.sh/uv/install.sh | sh'],check=True)\n"
            "os.environ['PATH']='/root/.local/bin:'+os.environ['PATH']\n"
            "subprocess.run(['uv','sync','--frozen'],cwd=repo_dir,check=True)\n"
            "print('Stage 23D uses CPU and reuses Stage 23C parquet artifacts')\n"
        ),
        markdown(
            "## Stage 23C artifacts\n\n"
            "training OOF 77 cutsとdesign validation 62 cutsのposterior特徴をそのまま使います。\n"
        ),
        code(
            "stage23c_run=artifact_dir/'stage23c_oof_decoder_full_v001'\n"
            "required=[stage23c_run/'summary.json',stage23c_run/'training_oof_decoder_rows.parquet',"
            "stage23c_run/'validation_decoder_rows.parquet']\n"
            "for path in required: assert path.is_file(),path\n"
            "stage23c_summary=json.loads((stage23c_run/'summary.json').read_text())\n"
            "assert stage23c_summary['stage23c_complete'] and not stage23c_summary['promoted_to_stage23d']\n"
            "print(*required,sep='\\n')\n"
        ),
        markdown(
            "## Nested selection and design validation\n\n"
            "4つの事前固定profileをtraining OOF内でnested比較します。eligible profileがなければ"
            "validation予測はbaseと同一になり、自動的に不合格です。\n"
        ),
        code(
            "RUN_ID='stage23d_hierarchical_decoder_full_v001'; run_dir=artifact_dir/RUN_ID\n"
            "if run_dir.exists() and not (run_dir/'summary.json').is_file():\n"
            "    resolved=run_dir.resolve(); expected=(artifact_dir/RUN_ID).resolve()\n"
            "    assert resolved==expected and resolved.parent==artifact_dir.resolve(),resolved\n"
            "    print('Removing incomplete prior run:',resolved); shutil.rmtree(resolved)\n"
            "if not (run_dir/'summary.json').is_file():\n"
            "    command=['uv','run','rogii-emission-hierarchical-decoder','--config',"
            "'configs/experiment/stage23d_hierarchical_decoder.yaml','--stage23c-run',str(stage23c_run),"
            "'--artifact-dir',str(artifact_dir),'--run-id',RUN_ID]\n"
            "    result=subprocess.run(command,cwd=repo_dir,text=True,capture_output=True)\n"
            "    if result.stdout: print(result.stdout)\n"
            "    if result.returncode:\n"
            "        print(result.stderr); raise RuntimeError(f'Stage 23D failed: {command}')\n"
            "summary=json.loads((run_dir/'summary.json').read_text())\n"
            "{key:summary[key] for key in ['stage23d_complete','promoted_to_stage23e_disjoint_confirmation',"
            "'training_cuts','training_wells','validation_cuts','validation_wells',"
            "'training_validation_well_overlap','selected_profile','validation_base_rmse',"
            "'validation_candidate_rmse','validation_delta','validation_p90_delta',"
            "'validation_bootstrap_95pct','validation_role','gates','next_step']}\n"
        ),
        code(
            "import pandas as pd\n"
            "pd.DataFrame(summary['nested_profile_report']).sort_values(['eligible','rmse'],ascending=[False,True])\n"
        ),
        markdown(
            "最後のsummary辞書とnested profile表を共有してください。通過しても、この結果から"
            "profileを差し替えたりKaggle提出を作ったりしません。\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "CPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "stage23d": {
                "submission": False,
                "standalone_setup": True,
                "gpu_required": False,
                "stage23c_volumes_reused": True,
                "validation_role": "design_validation",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
