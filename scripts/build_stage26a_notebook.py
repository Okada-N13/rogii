"""Build the standalone Stage 26A affine path-state audit notebook."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("notebooks/630_run_stage26a_affine_path_state.ipynb")


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
            "# Stage 26A: affine trajectory path-state signal audit\n\n"
            "rowwise offset stateを終了し、cut全体を開始offset×終了offsetのaffine trajectoryとして"
            "表現します。11×11=121 pathsをGR cost volume上で採点し、oracle headroomとraw rank信号を"
            "監査します。学習・Kaggle提出・予約120 wellsの使用はありません。CPUで実行できます。\n"
        ),
        code("from google.colab import drive\ndrive.mount('/content/drive')\n"),
        code(
            "from pathlib import Path\nimport json,os,shutil,subprocess,sys\n"
            "REPOSITORY_URL='https://github.com/Okada-N13/rogii.git'\n"
            "repo_dir=Path('/content/ROGII'); drive_root=Path('/content/drive/MyDrive/kaggle/rogii')\n"
            "artifact_dir=drive_root/'artifacts'; data_dir=drive_root/'data'\n"
            "if not (repo_dir/'.git').is_dir(): subprocess.run(['git','clone',REPOSITORY_URL,str(repo_dir)],check=True)\n"
            "else: subprocess.run(['git','-C',str(repo_dir),'pull','--ff-only','origin','main'],check=True)\n"
            "if shutil.which('uv') is None: subprocess.run(['bash','-lc','curl -LsSf https://astral.sh/uv/install.sh | sh'],check=True)\n"
            "os.environ['PATH']='/root/.local/bin:'+os.environ['PATH']\n"
            "subprocess.run(['uv','sync','--frozen'],cwd=repo_dir,check=True)\n"
            "assert (data_dir/'train').is_dir(),data_dir\n"
            "print('Stage 26A uses CPU; no TCN training or inference')\n"
        ),
        code(
            "stage16b_run=artifact_dir/'stage16b_testlike_validation_full_v003'\n"
            "stage17a_run=artifact_dir/'stage17_public_replay_full_v002'\n"
            "public_oof_run=artifact_dir/'stage7_public_residual_gate_full_v001'\n"
            "stage21b_run=artifact_dir/'stage21b_prefix_confidence_full_v001'\n"
            "stage24a_run=artifact_dir/'stage24a_scaled_ordinal_emission_full_v001'\n"
            "required=[stage16b_run/'well_assignments.parquet',stage17a_run/'cut_report.parquet',"
            "public_oof_run/'base_oof.parquet',stage21b_run/'confidence_cut_report.parquet',"
            "stage24a_run/'summary.json']\n"
            "for path in required: assert path.is_file(),path\n"
            "print(*required,sep='\\n')\n"
        ),
        markdown(
            "## 121 affine path states\n\n"
            "開始/終了offsetを-20..20 ft、4 ft刻みにし、raw GR costのoracle rankを評価します。\n"
        ),
        code(
            "RUN_ID='stage26a_affine_path_state_full_v001'; run_dir=artifact_dir/RUN_ID\n"
            "if run_dir.exists() and not (run_dir/'summary.json').is_file():\n"
            "    resolved=run_dir.resolve(); expected=(artifact_dir/RUN_ID).resolve()\n"
            "    assert resolved==expected and resolved.parent==artifact_dir.resolve(),resolved\n"
            "    print('Removing incomplete prior run:',resolved); shutil.rmtree(resolved)\n"
            "if not (run_dir/'summary.json').is_file():\n"
            "    command=[sys.executable,'-m','rogii.cli.affine_path_state','--config',"
            "'configs/experiment/stage26a_affine_path_state.yaml','--stage16b-run',str(stage16b_run),"
            "'--stage17a-run',str(stage17a_run),'--public-oof-run',str(public_oof_run),"
            "'--stage24a-run',str(stage24a_run),'--validation-run',str(stage21b_run),"
            "'--data-dir',str(data_dir),'--artifact-dir',str(artifact_dir),'--run-id',RUN_ID]\n"
            "    audit_env=os.environ.copy()\n"
            "    audit_env['PYTHONPATH']=str(repo_dir/'src')+':'+audit_env.get('PYTHONPATH','')\n"
            "    result=subprocess.run(command,cwd=repo_dir,env=audit_env,text=True,capture_output=True)\n"
            "    if result.stdout: print(result.stdout)\n"
            "    if result.stderr: print(result.stderr)\n"
            "    if result.returncode: raise RuntimeError(f'Stage 26A failed: {command}')\n"
            "summary=json.loads((run_dir/'summary.json').read_text())\n"
            "{key:summary[key] for key in ['stage26a_complete','promoted_to_stage26b_learned_path_ranker',"
            "'cuts','wells','rows','endpoint_states','affine_path_states','base_rmse','oracle_rmse',"
            "'oracle_delta','mean_valid_path_fraction','median_oracle_rank','top5_recall','top10_recall',"
            "'random_top10_recall','gates','reserved_confirmation_used','next_step']}\n"
        ),
        code(
            "import pandas as pd\n"
            "pd.DataFrame(summary['decoder_report']).sort_values('rmse')\n"
        ),
        markdown(
            "summary辞書とdecoder表を共有してください。昇格判定はdecoder RMSEではなく、"
            "事前固定したoracle/rank/group gateで行います。\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "CPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "stage26a": {
                "submission": False,
                "standalone_setup": True,
                "training": False,
                "reserved_confirmation_used": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
