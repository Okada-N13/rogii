"""Build the standalone Stage 23A strong-base physical-state audit notebook."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("notebooks/570_run_stage23a_strong_base_ncc.ipynb")


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
            "# Stage 23A: strong-base aligned physical offset-state audit\n\n"
            "Stage 22のrowwise residual fieldは終了します。A130 baseの周囲に-30〜+30 ftの"
            "離散offset状態を置き、horizontal GRとtypewell GRのraw multi-scale NCCが真の"
            "補正状態を識別できるかをStage 21Bの非重複58 wellsで監査します。oracle、top-k、"
            "全fold family、固定smooth decoderを測ります。学習もKaggle提出も行いません。"
            "CPUランタイムを使用してください。\n"
        ),
        code("from google.colab import drive\ndrive.mount('/content/drive')\n"),
        code(
            "from pathlib import Path\nimport json,os,shutil,subprocess\n"
            "REPOSITORY_URL='https://github.com/Okada-N13/rogii.git'\n"
            "repo_dir=Path('/content/ROGII'); drive_root=Path('/content/drive/MyDrive/kaggle/rogii')\n"
            "artifact_dir=drive_root/'artifacts'; data_dir=drive_root/'data'\n"
            "if not (repo_dir/'.git').is_dir(): subprocess.run(['git','clone',REPOSITORY_URL,str(repo_dir)],check=True)\n"
            "else: subprocess.run(['git','-C',str(repo_dir),'pull','--ff-only','origin','main'],check=True)\n"
            "if shutil.which('uv') is None: subprocess.run(['bash','-lc','curl -LsSf https://astral.sh/uv/install.sh | sh'],check=True)\n"
            "os.environ['PATH']='/root/.local/bin:'+os.environ['PATH']\n"
            "subprocess.run(['uv','sync','--frozen'],cwd=repo_dir,check=True)\n"
            "assert (data_dir/'train').is_dir(),data_dir\n"
            "def run_checked(command):\n"
            "    result=subprocess.run(command,cwd=repo_dir,text=True,capture_output=True)\n"
            "    if result.stdout: print(result.stdout,flush=True)\n"
            "    if result.returncode:\n"
            "        print(result.stderr,flush=True); raise RuntimeError(f'command failed: {command}')\n"
        ),
        markdown(
            "## 固定validation split\n\nStage 21Bの62 cuts・58 wellsだけを使います。"
            "offset stateの真値は評価とoracleにだけ使い、NCC emissionとdecoderはsuffix TVTを読みません。\n"
        ),
        code(
            "stage16b_run=artifact_dir/'stage16b_testlike_validation_full_v003'\n"
            "stage17a_run=artifact_dir/'stage17_public_replay_full_v002'\n"
            "public_oof_run=artifact_dir/'stage7_public_residual_gate_full_v001'\n"
            "stage21b_run=artifact_dir/'stage21b_prefix_confidence_full_v001'\n"
            "required=[stage16b_run/'well_assignments.parquet',stage17a_run/'cut_report.parquet',"
            "public_oof_run/'base_oof.parquet',stage21b_run/'confidence_cut_report.parquet']\n"
            "for path in required: assert path.is_file(),path\n"
            "print(*required,sep='\\n')\n"
        ),
        markdown(
            "## Raw emission and fixed decoder audit\n\n各cutでA130 baseを一度生成し、61 offset statesの"
            "NCCを計算します。10 cutsごとに進捗を表示します。途中失敗時はこのrunだけを安全に削除します。\n"
        ),
        code(
            "RUN_ID='stage23a_strong_base_ncc_full_v001'; run_dir=artifact_dir/RUN_ID\n"
            "if run_dir.exists() and not (run_dir/'summary.json').is_file():\n"
            "    resolved=run_dir.resolve(); expected=(artifact_dir/RUN_ID).resolve()\n"
            "    assert resolved==expected and resolved.parent==artifact_dir.resolve(),resolved\n"
            "    print('Removing incomplete prior run:',resolved); shutil.rmtree(resolved)\n"
            "if not (run_dir/'summary.json').is_file():\n"
            "    run_checked(['uv','run','rogii-strong-base-ncc','--config',"
            "'configs/experiment/stage23a_strong_base_ncc.yaml','--stage16b-run',str(stage16b_run),"
            "'--stage17a-run',str(stage17a_run),'--public-oof-run',str(public_oof_run),"
            "'--validation-run',str(stage21b_run),'--data-dir',str(data_dir),"
            "'--artifact-dir',str(artifact_dir),'--run-id',RUN_ID])\n"
            "summary=json.loads((run_dir/'summary.json').read_text())\n"
            "{key:summary[key] for key in ['stage23a_complete','promoted_to_stage23b_learned_emission',"
            "'cuts','wells','rows','offset_states','offset_coverage','emission_valid_fraction',"
            "'surface_rmse','oracle_rmse','oracle_delta','median_true_state_rank','top5_recall',"
            "'top10_recall','random_top10_recall','signal_fold_counts','fraction_signal_count',"
            "'gates','next_step']}\n"
        ),
        code(
            "import pandas as pd\n"
            "pd.DataFrame(summary['profile_report']).sort_values('rmse')\n"
        ),
        markdown(
            "最後のsummary辞書とprofile表を共有してください。raw rank信号が全groupで安定した場合だけ"
            "Stage 23B learned emissionへ進みます。decoderの最良profileを後付け採用しません。\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "CPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "stage23a": {
                "submission": False,
                "standalone_setup": True,
                "learned_model": False,
                "physical_state": "strong_base_tvt_offset",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
