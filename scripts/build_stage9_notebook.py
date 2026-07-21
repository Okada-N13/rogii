from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("notebooks/160_run_stage8_safe_cutback_gate.ipynb")
OUTPUT = Path("notebooks/180_run_stage9_residual_tcn.ipynb")


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


source = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = [
    markdown("""# Stage 9A: independent residual TCN

This standalone Colab notebook trains a new lightweight TCN against the verified ravaghi OOF residual. Features are rebuilt only from competition inputs, the fixed base prediction, and prefix-derived diagnostics. Stage 9A performs standard five-fold cross-fit only; it never submits to Kaggle.
"""),
    source["cells"][1],
    source["cells"][2],
    source["cells"][3],
    source["cells"][4],
    source["cells"][5],
    source["cells"][6],
    markdown("""## GPU check

Use a T4 or P100 runtime. T4 is preferred for mixed-precision Conv1d training. The Colab kernel's preinstalled PyTorch is used directly; it is not downloaded into the uv environment.
"""),
    code("""import importlib.util, sys
required = ['torch', 'pandas', 'pyarrow', 'sklearn', 'yaml']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyarrow', 'scikit-learn', 'pyyaml'], check=True)
import torch
assert torch.cuda.is_available(), 'Change the Colab runtime to a T4 or P100 GPU and reconnect.'
print('Python:', sys.executable)
print('PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0))
"""),
    markdown("""## Train five cross-fit TCN models

Keep `LIMIT_WELLS = None` for the decision run. Checkpoints and logs are written to Google Drive after every fold.
"""),
    code("""RUN_ID = 'stage9_residual_tcn_full_v001'
LIMIT_WELLS = None
cutback_run = artifact_dir / 'stage8_safe_cutback_gate_full_v002'
assert (cutback_run / 'candidate_matrix.parquet').is_file(), cutback_run
run_dir = artifact_dir / RUN_ID
if not (run_dir / 'gate_summary.json').is_file():
    command = [
        sys.executable, '-m', 'rogii.cli.sequence',
        '--config', 'configs/experiment/stage9_residual_tcn.yaml',
        '--cutback-run', str(cutback_run),
        '--data-dir', str(data_dir),
        '--artifact-dir', str(artifact_dir),
        '--run-id', RUN_ID,
    ]
    if LIMIT_WELLS is not None:
        command += ['--limit-wells', str(LIMIT_WELLS)]
    sequence_env = os.environ.copy()
    sequence_env['PYTHONPATH'] = str(repo_dir / 'src') + ':' + sequence_env.get('PYTHONPATH', '')
    log_path = artifact_dir / f'{RUN_ID}_driver.log'
    with log_path.open('w', encoding='utf-8') as log_handle:
        process = subprocess.Popen(
            command, cwd=repo_dir, env=sequence_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end='')
            log_handle.write(line); log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        tail = log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-100:]
        print('\\n'.join(tail))
        raise RuntimeError(f'Stage 9 failed with exit code {return_code}. Full log: {log_path}')
else:
    print('Reusing completed run:', run_dir)
"""),
    code("""gate = json.loads((run_dir / 'gate_summary.json').read_text())
{
    'promoted_to_spatial_audit': gate['promoted_to_spatial_audit'],
    'base_rmse': gate['base_metrics']['pooled_rmse'],
    'nested_candidate_rmse': gate['candidate_metrics']['pooled_rmse'],
    'rmse_delta': gate['pooled_rmse_delta'],
    'bootstrap_95pct': [gate['bootstrap']['ci_2_5'], gate['bootstrap']['ci_97_5']],
    'improved_folds': f"{gate['improved_folds']}/{len(gate['fold_deltas'])}",
    'fold_deltas': gate['fold_deltas'],
    'gates': gate['gates'],
    'inference_weight': gate['inference_weight'],
    'selections': gate['selections'],
    'weight_report': gate['weight_report'],
    'promotion_note': gate['promotion_note'],
}
"""),
    markdown("""## Decision

`promoted_to_spatial_audit: True` authorizes Stage 9B spatial cross-fit, not Kaggle submission. A false result closes this TCN design without spending a Kaggle run.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "accelerator": "GPU",
        "stage9": {"submission": False, "spatial_audit_required": True, "hidden_target_invariance": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
