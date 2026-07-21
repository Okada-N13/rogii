from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("notebooks/160_run_stage8_safe_cutback_gate.ipynb")


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


cells = [
    markdown("""# Stage 8A: Safe MHA visible-prefix cutback gate

This standalone Colab notebook audits the Gold visible-prefix physics layer already used by the 6.997 Safe MHA. It performs nested standard/spatial validation and does not create a Kaggle submission. The fleongg weight is intentionally fixed because that public package has no verified OOF predictions.
"""),
    code("""from google.colab import drive
drive.mount('/content/drive')
"""),
    code("""from pathlib import Path
import json, os, shutil, subprocess

REPOSITORY_URL = 'https://github.com/Okada-N13/rogii.git'
repo_dir = Path('/content/ROGII')
drive_root = Path('/content/drive/MyDrive/kaggle/rogii')
data_dir = drive_root / 'data'
artifact_dir = drive_root / 'artifacts'
assert (data_dir / 'train').is_dir(), f'Missing: {data_dir / "train"}'
if not (repo_dir / '.git').is_dir():
    subprocess.run(['git', 'clone', REPOSITORY_URL, str(repo_dir)], check=True)
else:
    subprocess.run(['git', '-C', str(repo_dir), 'pull', '--ff-only', 'origin', 'main'], check=True)
if shutil.which('uv') is None:
    subprocess.run(['bash', '-lc', 'curl -LsSf https://astral.sh/uv/install.sh | sh'], check=True)
os.environ['PATH'] = '/root/.local/bin:' + os.environ['PATH']
subprocess.run(['uv', 'sync', '--frozen'], cwd=repo_dir, check=True)
python = repo_dir / '.venv/bin/python'
subprocess.run(['uv', 'pip', 'install', '--python', str(python), 'kagglehub', 'lightgbm', 'catboost'], check=True)
artifact_dir.mkdir(parents=True, exist_ok=True)
subprocess.run(['git', '-C', str(repo_dir), 'rev-parse', '--short', 'HEAD'], check=True)
"""),
    markdown("""## Strong-base preparation

The notebook reuses the verified ravaghi `base_oof.parquet` from Stage 7 when present. If it is missing, the following cells reconstruct it from the public artifact. This fallback is memory-heavy but makes Stage 8 standalone.
"""),
    code("""BASE_RUN_ID = 'stage7_public_residual_gate_full_v001'
base_run = artifact_dir / BASE_RUN_ID
if not (base_run / 'base_oof.parquet').is_file():
    download = subprocess.run([
        str(python), '-c',
        "import kagglehub; print(kagglehub.dataset_download('ravaghi/wellbore-geology-prediction-artifacts'))"
    ], check=True, capture_output=True, text=True)
    download_root = Path(download.stdout.strip().splitlines()[-1])
    train_candidates = list(download_root.rglob('data/train.csv'))
    assert len(train_candidates) == 1, train_candidates
    public_artifacts_dir = train_candidates[0].parent.parent
    probe = subprocess.run([str(python), '-c', 'import koolbox'], capture_output=True)
    if probe.returncode != 0:
        direct = subprocess.run(['uv', 'pip', 'install', '--python', str(python), 'koolbox==0.1.3'])
        if direct.returncode != 0:
            kb_download = subprocess.run([
                str(python), '-c',
                "import kagglehub; print(kagglehub.dataset_download('phongnguyn23021656/koolbox-offline'))"
            ], check=True, capture_output=True, text=True)
            kb_root = Path(kb_download.stdout.strip().splitlines()[-1])
            for wheel in kb_root.rglob('*.whl'):
                subprocess.run(['uv', 'pip', 'install', '--python', str(python), '--no-deps', str(wheel)], check=True)
    subprocess.run([
        'uv', 'run', 'rogii-public-oof',
        '--config', 'configs/experiment/public_residual_gate.yaml',
        '--public-artifacts-dir', str(public_artifacts_dir),
        '--data-dir', str(data_dir),
        '--artifact-dir', str(artifact_dir),
        '--run-id', BASE_RUN_ID,
    ], cwd=repo_dir, check=True)
else:
    print('Reusing strong-base OOF:', base_run)
"""),
    markdown("""## Run Stage 8A

Keep `LIMIT_WELLS = None` for the promotion decision. Use 10 or more only for a wiring smoke test.
"""),
    code("""RUN_ID = 'stage8_safe_cutback_gate_full_v001'
LIMIT_WELLS = None
run_dir = artifact_dir / RUN_ID
if not (run_dir / 'gate_summary.json').is_file():
    command = [
        'uv', 'run', 'rogii-safe-cutback',
        '--config', 'configs/experiment/stage8_safe_cutback_gate.yaml',
        '--base-run', str(base_run),
        '--data-dir', str(data_dir),
        '--artifact-dir', str(artifact_dir),
        '--run-id', RUN_ID,
    ]
    if LIMIT_WELLS is not None:
        command += ['--limit-wells', str(LIMIT_WELLS)]
    subprocess.run(command, cwd=repo_dir, check=True)
else:
    print('Reusing completed run:', run_dir)
"""),
    code("""gate = json.loads((run_dir / 'gate_summary.json').read_text())
{
    'promoted': gate['promoted'],
    'base_rmse': gate['base_metrics']['pooled_rmse'],
    'nested_candidate_rmse': gate['candidate_metrics']['pooled_rmse'],
    'rmse_delta': gate['pooled_rmse_delta'],
    'bootstrap_95pct': [gate['bootstrap']['ci_2_5'], gate['bootstrap']['ci_97_5']],
    'improved_folds': f"{gate['improved_folds']}/{len(gate['fold_deltas'])}",
    'gates': gate['gates'],
    'spatial_delta': gate['spatial']['pooled_rmse_delta'],
    'spatial_fold_deltas': gate['spatial']['fold_deltas'],
    'inference_profile': gate['inference_profile'],
    'standard_selections': gate['standard_selections'],
    'spatial_selections': gate['spatial_selections'],
    'profile_metrics': gate['profile_metrics'],
}
"""),
    markdown("""## Decision

Only `promoted: True` with a non-null `inference_profile` authorizes a Kaggle integration. A false result means the existing Safe MHA Gold layer stays unchanged; do not spend an eight-hour Kaggle run.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "stage8": {"submission": False, "standalone": True, "base": "ravaghi_verified_oof"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
