"""Build the standalone Stage 23B learned strong-base emission notebook."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("notebooks/580_run_stage23b_learned_emission.ipynb")


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
            "# Stage 23B: disjoint learned strong-base emission\n\n"
            "Stage 23Aで確認したstrong-base offset-state信号をTCNで学習します。Stage 21Aの"
            "77 cuts・63 wellsだけをtrainingに使い、Stage 21Bの62 cuts・58 wellsは固定外部検証"
            "にだけ使います。training内5-foldモデルのlogit ensembleで評価し、validation targetを"
            "early stoppingやモデル選択に使いません。Kaggle提出は作りません。T4 GPUを使用してください。\n"
        ),
        code("from google.colab import drive\ndrive.mount('/content/drive')\n"),
        code(
            "from pathlib import Path\nimport importlib.util,json,os,shutil,subprocess,sys\n"
            "REPOSITORY_URL='https://github.com/Okada-N13/rogii.git'\n"
            "repo_dir=Path('/content/ROGII'); drive_root=Path('/content/drive/MyDrive/kaggle/rogii')\n"
            "artifact_dir=drive_root/'artifacts'; data_dir=drive_root/'data'\n"
            "if not (repo_dir/'.git').is_dir(): subprocess.run(['git','clone',REPOSITORY_URL,str(repo_dir)],check=True)\n"
            "else: subprocess.run(['git','-C',str(repo_dir),'pull','--ff-only','origin','main'],check=True)\n"
            "if shutil.which('uv') is None: subprocess.run(['bash','-lc','curl -LsSf https://astral.sh/uv/install.sh | sh'],check=True)\n"
            "os.environ['PATH']='/root/.local/bin:'+os.environ['PATH']\n"
            "subprocess.run(['uv','sync','--frozen'],cwd=repo_dir,check=True)\n"
            "import torch\n"
            "assert torch.cuda.is_available(),'Select a GPU runtime and reconnect'\n"
            "assert (data_dir/'train').is_dir(),data_dir\n"
            "print('GPU:',torch.cuda.get_device_name(0),'PyTorch:',torch.__version__)\n"
        ),
        markdown(
            "## 固定splitとStage 23A gate\n\nStage 23Aの全gate通過を確認します。"
            "Stage 21A/21Bのwell overlapはCLIでも再検査します。\n"
        ),
        code(
            "stage16b_run=artifact_dir/'stage16b_testlike_validation_full_v003'\n"
            "stage17a_run=artifact_dir/'stage17_public_replay_full_v002'\n"
            "public_oof_run=artifact_dir/'stage7_public_residual_gate_full_v001'\n"
            "stage21a_run=artifact_dir/'stage21a_prefix_router_full_v001'\n"
            "stage21b_run=artifact_dir/'stage21b_prefix_confidence_full_v001'\n"
            "stage23a_run=artifact_dir/'stage23a_strong_base_ncc_full_v001'\n"
            "required=[stage16b_run/'well_assignments.parquet',stage17a_run/'cut_report.parquet',"
            "public_oof_run/'base_oof.parquet',stage21a_run/'router_cut_report.parquet',"
            "stage21b_run/'confidence_cut_report.parquet',stage23a_run/'summary.json']\n"
            "for path in required: assert path.is_file(),path\n"
            "print(*required,sep='\\n')\n"
        ),
        markdown(
            "## 5-model TCN ensemble\n\ncost volume構築後、training split内の5 foldsで学習します。"
            "T4 1枚を使用します。途中失敗時はこのrunだけを安全に削除して最初から実行します。\n"
        ),
        code(
            "RUN_ID='stage23b_learned_emission_full_v001'; run_dir=artifact_dir/RUN_ID\n"
            "if run_dir.exists() and not (run_dir/'summary.json').is_file():\n"
            "    resolved=run_dir.resolve(); expected=(artifact_dir/RUN_ID).resolve()\n"
            "    assert resolved==expected and resolved.parent==artifact_dir.resolve(),resolved\n"
            "    print('Removing incomplete prior run:',resolved); shutil.rmtree(resolved)\n"
            "if not (run_dir/'summary.json').is_file():\n"
            "    command=[sys.executable,'-m','rogii.cli.strong_base_emission','--config',"
            "'configs/experiment/stage23b_learned_strong_base_emission.yaml',"
            "'--stage16b-run',str(stage16b_run),'--stage17a-run',str(stage17a_run),"
            "'--public-oof-run',str(public_oof_run),'--stage23a-run',str(stage23a_run),"
            "'--training-run',str(stage21a_run),'--validation-run',str(stage21b_run),"
            "'--data-dir',str(data_dir),'--artifact-dir',str(artifact_dir),"
            "'--run-id',RUN_ID,'--device','cuda']\n"
            "    emission_env=os.environ.copy()\n"
            "    emission_env['PYTHONPATH']=str(repo_dir/'src')+':'+emission_env.get('PYTHONPATH','')\n"
            "    log_path=artifact_dir/f'{RUN_ID}_driver.log'\n"
            "    with log_path.open('w',encoding='utf-8') as log_handle:\n"
            "        process=subprocess.Popen(command,cwd=repo_dir,env=emission_env,"
            "stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n"
            "        for line in process.stdout:\n"
            "            print(line,end=''); log_handle.write(line); log_handle.flush()\n"
            "        return_code=process.wait()\n"
            "    if return_code!=0:\n"
            "        print('\\n'.join(log_path.read_text(encoding='utf-8',errors='replace').splitlines()[-160:]))\n"
            "        raise RuntimeError(f'Stage 23B failed with exit code {return_code}. Full log: {log_path}')\n"
            "summary=json.loads((run_dir/'summary.json').read_text())\n"
            "{key:summary[key] for key in ['stage23b_complete','promoted_to_stage23c','device',"
            "'training_cuts','training_wells','validation_cuts','validation_wells',"
            "'training_validation_well_overlap','ensemble_models','rows','raw','learned',"
            "'top10_gain','top5_gain','nll_delta','base_rmse','candidate_rmse','rmse_delta',"
            "'well_p90_delta','bootstrap_95pct','group_improved_counts','gates','next_step']}\n"
        ),
        code(
            "import pandas as pd\n"
            "pd.DataFrame(summary['weight_report']).sort_values('weight')\n"
        ),
        markdown(
            "最後のsummary辞書とweight表を共有してください。rankだけでなく固定weight 0.50の"
            "RMSE・bootstrap・P90も通った場合だけStage 23Cへ進みます。\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "stage23b": {
                "submission": False,
                "standalone_setup": True,
                "accelerator": "T4",
                "external_validation_target_used_for_training": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
