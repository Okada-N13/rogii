"""Build the standalone Stage 23C nested OOF decoder notebook."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("notebooks/590_run_stage23c_oof_decoder.ipynb")


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
            "# Stage 23C: nested OOF emission decoder\n\n"
            "Stage 23BのTCN rankerは維持し、posteriorからoffsetへ変換するdecoderだけを修正します。"
            "Stage 21A training OOF logits上でdecoderをnested cross-fitし、全selection gateを通った"
            "profileだけを全training OOFでfitします。その後に初めてStage 21B validationへ適用します。"
            "validation targetはprofile選択に使いません。Kaggle提出は作りません。T4/L4 GPUを使用してください。\n"
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
            "import torch\n"
            "assert torch.cuda.is_available(),'Select a GPU runtime and reconnect'\n"
            "assert (data_dir/'train').is_dir(),data_dir\n"
            "print('GPU:',torch.cuda.get_device_name(0),'PyTorch:',torch.__version__)\n"
        ),
        markdown(
            "## 固定artifacts\n\nStage 23Bの5 checkpointsを再利用します。TCNは再学習しません。"
            "Stage 21A/21Bのsplitも変更しません。\n"
        ),
        code(
            "stage16b_run=artifact_dir/'stage16b_testlike_validation_full_v003'\n"
            "stage17a_run=artifact_dir/'stage17_public_replay_full_v002'\n"
            "public_oof_run=artifact_dir/'stage7_public_residual_gate_full_v001'\n"
            "stage21a_run=artifact_dir/'stage21a_prefix_router_full_v001'\n"
            "stage21b_run=artifact_dir/'stage21b_prefix_confidence_full_v001'\n"
            "stage23b_run=artifact_dir/'stage23b_learned_emission_full_v001'\n"
            "required=[stage16b_run/'well_assignments.parquet',stage17a_run/'cut_report.parquet',"
            "public_oof_run/'base_oof.parquet',stage21a_run/'router_cut_report.parquet',"
            "stage21b_run/'confidence_cut_report.parquet',stage23b_run/'summary.json']\n"
            "required += [stage23b_run/f'fold_{fold}.pt' for fold in range(5)]\n"
            "for path in required: assert path.is_file(),path\n"
            "print(*required,sep='\\n')\n"
        ),
        markdown(
            "## Nested decoder selection and untouched validation\n\ntraining OOFでdirect/affine/summary ridgeを"
            "nested比較します。選択profileがなければvalidationはbaseのままになり、不合格になります。\n"
        ),
        code(
            "RUN_ID='stage23c_oof_decoder_full_v001'; run_dir=artifact_dir/RUN_ID\n"
            "if run_dir.exists() and not (run_dir/'summary.json').is_file():\n"
            "    resolved=run_dir.resolve(); expected=(artifact_dir/RUN_ID).resolve()\n"
            "    assert resolved==expected and resolved.parent==artifact_dir.resolve(),resolved\n"
            "    print('Removing incomplete prior run:',resolved); shutil.rmtree(resolved)\n"
            "if not (run_dir/'summary.json').is_file():\n"
            "    command=[sys.executable,'-m','rogii.cli.emission_decoder','--config',"
            "'configs/experiment/stage23c_oof_decoder.yaml','--stage16b-run',str(stage16b_run),"
            "'--stage17a-run',str(stage17a_run),'--public-oof-run',str(public_oof_run),"
            "'--stage23b-run',str(stage23b_run),'--training-run',str(stage21a_run),"
            "'--validation-run',str(stage21b_run),'--data-dir',str(data_dir),"
            "'--artifact-dir',str(artifact_dir),'--run-id',RUN_ID,'--device','cuda']\n"
            "    decoder_env=os.environ.copy()\n"
            "    decoder_env['PYTHONPATH']=str(repo_dir/'src')+':'+decoder_env.get('PYTHONPATH','')\n"
            "    log_path=artifact_dir/f'{RUN_ID}_driver.log'\n"
            "    with log_path.open('w',encoding='utf-8') as log_handle:\n"
            "        process=subprocess.Popen(command,cwd=repo_dir,env=decoder_env,"
            "stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n"
            "        for line in process.stdout:\n"
            "            print(line,end=''); log_handle.write(line); log_handle.flush()\n"
            "        return_code=process.wait()\n"
            "    if return_code!=0:\n"
            "        print('\\n'.join(log_path.read_text(encoding='utf-8',errors='replace').splitlines()[-160:]))\n"
            "        raise RuntimeError(f'Stage 23C failed with exit code {return_code}. Full log: {log_path}')\n"
            "summary=json.loads((run_dir/'summary.json').read_text())\n"
            "{key:summary[key] for key in ['stage23c_complete','promoted_to_stage23d','device',"
            "'training_cuts','training_wells','validation_cuts','validation_wells',"
            "'training_validation_well_overlap','selected_profile','validation_base_rmse',"
            "'validation_candidate_rmse','validation_delta','validation_p90_delta',"
            "'validation_bootstrap_95pct','gates','next_step']}\n"
        ),
        code(
            "import pandas as pd\n"
            "pd.DataFrame(summary['nested_profile_report']).sort_values(['eligible','rmse'],ascending=[False,True])\n"
        ),
        markdown(
            "最後のsummary辞書とnested profile表を共有してください。Stage 21B validation結果を見て"
            "別profileへ差し替えることはしません。\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "stage23c": {
                "submission": False,
                "standalone_setup": True,
                "tcn_retraining": False,
                "validation_target_used_for_selection": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
