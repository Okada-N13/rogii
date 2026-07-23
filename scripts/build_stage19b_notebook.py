"""Build the standalone Colab notebook for Stage 19B."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("notebooks/490_build_stage19b_trajectory_package.ipynb")


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
            "# Stage 19B: all-data trajectory model bundle\n\n"
            "Stage 19Aで全gateを通過した3係数trajectory residualを全3,865 cutsで学習します。"
            "5 seeds × 3係数の小型木モデルをbundle化し、773坑井でtarget-free特徴再計算の一致と"
            "hidden約200坑井の推定追加時間を測ります。まだKaggle提出は行いません。\n"
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
        markdown("## 固定artifact\n\nStage 19A結果とStage 17 strong-base predictionを使います。\n"),
        code(
            "stage19a_run = artifact_dir / 'stage19a_trajectory_residual_full_v001'\n"
            "stage17a_run = artifact_dir / 'stage17_public_replay_full_v002'\n"
            "stage17b_run = artifact_dir / 'stage17b_selector_replay_full_v001'\n"
            "assert (stage19a_run / 'summary.json').is_file(), stage19a_run\n"
            "assert json.loads((stage19a_run / 'summary.json').read_text())['promoted_to_stage19b'] is True\n"
            "assert (stage17a_run / 'replay_predictions.parquet').is_file(), stage17a_run\n"
            "assert (stage17b_run / 'selector_predictions.parquet').is_file(), stage17b_run\n"
            "print(stage19a_run, stage17a_run, stage17b_run, sep='\\n')\n"
        ),
        markdown(
            "## 全data学習とruntime benchmark\n\nCPUで実行できます。bundleにはdonor CSV、KD-tree、"
            "rowwise neural modelを含めません。特徴再計算がStage 19A保存値と一致しない場合は停止します。\n"
        ),
        code(
            "RUN_ID = 'stage19b_trajectory_package_full_v001'\n"
            "run_dir = artifact_dir / RUN_ID\n"
            "if not (run_dir / 'summary.json').is_file():\n"
            "    run_checked([\n"
            "        'uv', 'run', 'rogii-trajectory-package',\n"
            "        '--config', 'configs/experiment/stage19b_trajectory_package.yaml',\n"
            "        '--stage19a-run', str(stage19a_run), '--stage17a-run', str(stage17a_run),\n"
            "        '--stage17b-run', str(stage17b_run), '--data-dir', str(data_dir),\n"
            "        '--artifact-dir', str(artifact_dir), '--run-id', RUN_ID,\n"
            "    ])\n"
            "else:\n"
            "    print('Reusing completed run:', run_dir)\n"
            "summary = json.loads((run_dir / 'summary.json').read_text())\n"
            "{\n"
            "    'stage19b_complete': summary['stage19b_complete'],\n"
            "    'promoted_to_stage19c': summary['promoted_to_stage19c'],\n"
            "    'training_cuts': summary['training_cuts'], 'training_wells': summary['training_wells'],\n"
            "    'feature_count': summary['feature_count'], 'ensemble_seeds': summary['ensemble_seeds'],\n"
            "    'model_count': summary['model_count'], 'benchmark': summary['benchmark'],\n"
            "    'package_manifest_sha256': summary['package_manifest_sha256'],\n"
            "    'zip': summary['zip'], 'gates': summary['gates'], 'next_step': summary['next_step'],\n"
            "}\n"
        ),
        markdown(
            "最後の辞書を共有してください。全gate通過時だけ、zipをKaggle Datasetへ上げる前に"
            "Stage 19C Internet-OFF推論コードと470統合Notebookを作ります。\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "CPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "stage19b": {
                "submission": False, "cpu_only": True, "standalone_setup": True,
                "all_data_models": True, "runtime_benchmark": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
