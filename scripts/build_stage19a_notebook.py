"""Build the standalone Colab notebook for Stage 19A."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("notebooks/480_run_stage19a_trajectory_residual.ipynb")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(keepends=True),
    }


def build() -> None:
    cells = [
        markdown(
            "# Stage 19A: cross-fitted low-dimensional trajectory residual\n\n"
            "実測6.589のtop-PF系を将来の提出baseとし、固定Stage 16B pseudo-test上では"
            "Stage 17 strong-base replayを代理baseとして使います。各cutのhidden tailを直接学習せず、"
            "滑らかな補正3係数だけをtarget-free特徴からcross-fitします。Kaggle提出はまだ行いません。\n"
        ),
        code("from google.colab import drive\ndrive.mount('/content/drive')\n"),
        code(
            "from pathlib import Path\n"
            "import json, os, shutil, subprocess\n"
            "REPOSITORY_URL = 'https://github.com/Okada-N13/rogii.git'\n"
            "repo_dir = Path('/content/ROGII')\n"
            "drive_root = Path('/content/drive/MyDrive/kaggle/rogii')\n"
            "artifact_dir = drive_root / 'artifacts'\n"
            "data_dir = drive_root / 'data'\n"
            "if not (repo_dir / '.git').is_dir():\n"
            "    subprocess.run(['git', 'clone', REPOSITORY_URL, str(repo_dir)], check=True)\n"
            "else:\n"
            "    subprocess.run(['git', '-C', str(repo_dir), 'pull', '--ff-only', 'origin', 'main'], check=True)\n"
            "if shutil.which('uv') is None:\n"
            "    subprocess.run(['bash', '-lc', 'curl -LsSf https://astral.sh/uv/install.sh | sh'], check=True)\n"
            "os.environ['PATH'] = '/root/.local/bin:' + os.environ['PATH']\n"
            "subprocess.run(['uv', 'sync', '--frozen'], cwd=repo_dir, check=True)\n"
            "assert (data_dir / 'train').is_dir(), f'Train data not found: {data_dir / \"train\"}'\n"
            "artifact_dir.mkdir(parents=True, exist_ok=True)\n"
            "def run_checked(command):\n"
            "    result = subprocess.run(command, cwd=repo_dir, text=True, capture_output=True)\n"
            "    if result.stdout:\n"
            "        print(result.stdout, flush=True)\n"
            "    if result.returncode != 0:\n"
            "        print(result.stderr, flush=True)\n"
            "        raise RuntimeError(f'Command failed with exit code {result.returncode}: {command}')\n"
            "    return result\n"
            "subprocess.run(['git', '-C', str(repo_dir), 'rev-parse', '--short', 'HEAD'], check=True)\n"
        ),
        markdown(
            "## 固定artifact\n\n"
            "Stage 16B v003のcut/foldと、Stage 17A/17Bのstrong-base予測を読みます。"
            "Stage 18のdonor cacheやKaggle packageは不要です。\n"
        ),
        code(
            "stage16b_run = artifact_dir / 'stage16b_testlike_validation_full_v003'\n"
            "stage17a_run = artifact_dir / 'stage17_public_replay_full_v002'\n"
            "stage17b_run = artifact_dir / 'stage17b_selector_replay_full_v001'\n"
            "assert (stage16b_run / 'well_assignments.parquet').is_file(), stage16b_run\n"
            "assert (stage17a_run / 'replay_predictions.parquet').is_file(), stage17a_run\n"
            "assert (stage17b_run / 'selector_predictions.parquet').is_file(), stage17b_run\n"
            "assert (stage17b_run / 'cut_report.parquet').is_file(), stage17b_run\n"
            "print(stage16b_run, stage17a_run, stage17b_run, sep='\\n')\n"
        ),
        markdown(
            "## Stage 19A cross-fit\n\n"
            "CPUで実行できます。18.8M行のbase predictionを必要な3列だけ読み、"
            "3,865 cutsを4 fold familyで評価します。通常のColab RAMで不足する場合だけ"
            "ハイメモリを選択してください。\n"
        ),
        code(
            "RUN_ID = 'stage19a_trajectory_residual_full_v001'\n"
            "run_dir = artifact_dir / RUN_ID\n"
            "if not (run_dir / 'summary.json').is_file():\n"
            "    run_checked([\n"
            "        'uv', 'run', 'rogii-trajectory-residual',\n"
            "        '--config', 'configs/experiment/stage19a_trajectory_residual.yaml',\n"
            "        '--stage16b-run', str(stage16b_run), '--stage17a-run', str(stage17a_run),\n"
            "        '--stage17b-run', str(stage17b_run), '--data-dir', str(data_dir),\n"
            "        '--artifact-dir', str(artifact_dir), '--run-id', RUN_ID,\n"
            "    ])\n"
            "else:\n"
            "    print('Reusing completed run:', run_dir)\n"
            "summary = json.loads((run_dir / 'summary.json').read_text())\n"
            "{\n"
            "    'stage19a_complete': summary['stage19a_complete'],\n"
            "    'promoted_to_stage19b': summary['promoted_to_stage19b'],\n"
            "    'cuts': summary['cuts'], 'wells': summary['wells'],\n"
            "    'profile': summary['profile'],\n"
            "    'standard': summary['family_reports']['fold'],\n"
            "    'spatial': summary['family_reports']['spatial_fold'],\n"
            "    'typewell': summary['family_reports']['typewell_fold'],\n"
            "    'branch_group': summary['family_reports']['branch_group_fold'],\n"
            "    'bootstrap_95pct': summary['bootstrap_95pct'],\n"
            "    'gates': summary['gates'], 'runtime_contract': summary['runtime_contract'],\n"
            "    'next_step': summary['next_step'],\n"
            "}\n"
        ),
        markdown("## 診断grid\n\n固定profile以外は昇格判定に使わず、次Stageの設計材料としてだけ表示します。\n"),
        code(
            "import pandas as pd\n"
            "profiles = pd.read_parquet(run_dir / 'profile_report.parquet')\n"
            "profiles[['weight', 'cap_ft', 'base_rmse', 'candidate_rmse', 'rmse_delta', "
            "'well_rmse_p90_delta', 'worst10_sse_share_delta']].sort_values('rmse_delta').head(16)\n"
        ),
        markdown(
            "最後の辞書と診断表を共有してください。4 family、tail、bootstrapをすべて通過した場合だけ、"
            "Stage 19Bで全data学習と10分以内のtest推論packageを作ります。\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "CPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "stage19a": {
                "submission": False, "cpu_only": True, "standalone_setup": True,
                "cross_fitted": True, "predicted_values_per_well": 3,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
