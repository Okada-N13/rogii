"""Build the standalone Stage 24A scaled ordinal emission notebook."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("notebooks/610_run_stage24a_scaled_ordinal_emission.ipynb")


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
            "# Stage 24A: scaled soft-ordinal emission\n\n"
            "Stage 23の77-cut rankerを500 wells・500 cutsへ拡大し、one-hot state分類を"
            "soft ordinal targetとexpected-offset Huber lossへ置き換えます。Stage 21Bの58 wellsは"
            "design validationとして学習から除外し、別の120 wellsをStage 24B確認用に予約します。"
            "Kaggle提出は作りません。A100またはL4推奨、T4でも実行可能です。\n"
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
            "## 固定split manifest\n\n"
            "foldごと24 reserved confirmation wellsをhashで固定し、trainingは合計500 wellsになるよう"
            "eligible wellsのfold間差を決定論的に再配分します。"
            "3集合のwell overlapが0であることを確認します。\n"
        ),
        code(
            "stage16b_run=artifact_dir/'stage16b_testlike_validation_full_v003'\n"
            "stage17a_run=artifact_dir/'stage17_public_replay_full_v002'\n"
            "public_oof_run=artifact_dir/'stage7_public_residual_gate_full_v001'\n"
            "stage21b_run=artifact_dir/'stage21b_prefix_confidence_full_v001'\n"
            "stage23a_run=artifact_dir/'stage23a_strong_base_ncc_full_v001'\n"
            "required=[stage16b_run/'well_assignments.parquet',stage17a_run/'cut_report.parquet',"
            "public_oof_run/'base_oof.parquet',stage21b_run/'confidence_cut_report.parquet',"
            "stage23a_run/'summary.json']\n"
            "for path in required: assert path.is_file(),path\n"
            "MANIFEST_ID='stage24a_scaled_emission_manifest_v003'; manifest_dir=artifact_dir/MANIFEST_ID\n"
            "if manifest_dir.exists() and not (manifest_dir/'summary.json').is_file():\n"
            "    resolved=manifest_dir.resolve(); expected=(artifact_dir/MANIFEST_ID).resolve()\n"
            "    assert resolved==expected and resolved.parent==artifact_dir.resolve(); shutil.rmtree(resolved)\n"
            "if not (manifest_dir/'summary.json').is_file():\n"
            "    manifest_command=['uv','run','rogii-scaled-emission-manifest',"
            "'--stage17a-run',str(stage17a_run),'--design-validation-run',str(stage21b_run),"
            "'--artifact-dir',str(artifact_dir),'--run-id',MANIFEST_ID]\n"
            "    result=subprocess.run(manifest_command,cwd=repo_dir,text=True,capture_output=True)\n"
            "    if result.stdout: print(result.stdout)\n"
            "    if result.stderr: print(result.stderr)\n"
            "    if result.returncode: raise RuntimeError(f'Manifest failed: {manifest_command}')\n"
            "manifest=json.loads((manifest_dir/'summary.json').read_text())\n"
            "assert manifest['public_replay_eligible_only'] is True,manifest\n"
            "assert manifest['training_wells']==500 and manifest['confirmation_wells']==120,manifest\n"
            "assert all(not values for values in manifest['overlaps'].values()),manifest['overlaps']\n"
            "manifest\n"
        ),
        markdown(
            "## 500-cut soft-ordinal training\n\n"
            "cost volume生成はCPU処理が中心で、その後5-fold TCNをGPU学習します。Driveへ逐次ログを"
            "保存します。中断時は不完全runだけを安全に削除して再実行します。\n"
        ),
        code(
            "RUN_ID='stage24a_scaled_ordinal_emission_full_v001'; run_dir=artifact_dir/RUN_ID\n"
            "if run_dir.exists() and not (run_dir/'summary.json').is_file():\n"
            "    resolved=run_dir.resolve(); expected=(artifact_dir/RUN_ID).resolve()\n"
            "    assert resolved==expected and resolved.parent==artifact_dir.resolve(),resolved\n"
            "    print('Removing incomplete prior run:',resolved); shutil.rmtree(resolved)\n"
            "if not (run_dir/'summary.json').is_file():\n"
            "    command=[sys.executable,'-m','rogii.cli.strong_base_emission','--config',"
            "'configs/experiment/stage24a_scaled_ordinal_emission.yaml','--stage16b-run',str(stage16b_run),"
            "'--stage17a-run',str(stage17a_run),'--public-oof-run',str(public_oof_run),"
            "'--stage23a-run',str(stage23a_run),'--training-cut-file',str(manifest_dir/'training_cut_ids.parquet'),"
            "'--validation-cut-file',str(stage21b_run/'confidence_cut_report.parquet'),"
            "'--data-dir',str(data_dir),'--artifact-dir',str(artifact_dir),'--run-id',RUN_ID,'--device','cuda']\n"
            "    training_env=os.environ.copy()\n"
            "    training_env['PYTHONPATH']=str(repo_dir/'src')+':'+training_env.get('PYTHONPATH','')\n"
            "    log_path=artifact_dir/f'{RUN_ID}_driver.log'\n"
            "    with log_path.open('w',encoding='utf-8') as log_handle:\n"
            "        process=subprocess.Popen(command,cwd=repo_dir,env=training_env,"
            "stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n"
            "        for line in process.stdout:\n"
            "            print(line,end=''); log_handle.write(line); log_handle.flush()\n"
            "        return_code=process.wait()\n"
            "    if return_code!=0:\n"
            "        print('\\n'.join(log_path.read_text(encoding='utf-8',errors='replace').splitlines()[-200:]))\n"
            "        raise RuntimeError(f'Stage 24A failed with exit code {return_code}. Full log: {log_path}')\n"
            "summary=json.loads((run_dir/'summary.json').read_text())\n"
            "{key:summary[key] for key in ['stage24a_complete','promoted_to_stage24b_reserved_confirmation',"
            "'device','training_cuts','training_wells','validation_cuts','validation_wells',"
            "'training_validation_well_overlap','ensemble_models','raw','learned','top10_gain','top5_gain',"
            "'nll_delta','base_rmse','candidate_rmse','rmse_delta','well_p90_delta',"
            "'bootstrap_95pct','group_improved_counts','gates','next_step']}\n"
        ),
        code(
            "import pandas as pd\n"
            "pd.DataFrame(summary['weight_report']).sort_values('weight')\n"
        ),
        markdown(
            "最後のsummary辞書、weight表、各foldのtraining logを共有してください。diagnostic weightから"
            "後付け選択せず、primary 0.75のgateだけで判断します。\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "stage24a": {
                "submission": False,
                "standalone_setup": True,
                "training_wells": 500,
                "reserved_confirmation_wells": 120,
                "validation_target_used_for_training": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
