from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("notebooks/160_run_stage8_safe_cutback_gate.ipynb")
OUTPUT = Path("notebooks/170_run_stage8b_conditional_well_gate.ipynb")


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
    markdown("""# Stage 8B: cross-fitted conditional well gate

Stage 8A found that the conservative physics profile was globally neutral. This standalone Colab notebook learns, strictly out of fold, which wells should receive that already-generated conservative correction. It does not submit to Kaggle.
"""),
    source["cells"][1],
    source["cells"][2],
    source["cells"][3],
    source["cells"][4],
    markdown("""## Prepare Stage 8A candidates

The completed v002 run is reused. If it is absent, the deterministic cutback candidate matrix is generated first.
"""),
    code("""CUTBACK_RUN_ID = 'stage8_safe_cutback_gate_full_v002'
cutback_run = artifact_dir / CUTBACK_RUN_ID
if not (cutback_run / 'candidate_matrix.parquet').is_file():
    subprocess.run([
        'uv', 'run', 'rogii-safe-cutback',
        '--config', 'configs/experiment/stage8_safe_cutback_gate.yaml',
        '--base-run', str(base_run),
        '--data-dir', str(data_dir),
        '--artifact-dir', str(artifact_dir),
        '--run-id', CUTBACK_RUN_ID,
    ], cwd=repo_dir, check=True)
else:
    print('Reusing Stage 8A:', cutback_run)
"""),
    markdown("""## Run the nested conditional gate

The model sees only prefix/candidate diagnostics. Improvement labels are used only inside training folds. Thresholds are selected from inner OOF predictions before application to each outer fold.
"""),
    code("""RUN_ID = 'stage8b_conditional_well_gate_full_v001'
run_dir = artifact_dir / RUN_ID
if not (run_dir / 'gate_summary.json').is_file():
    command = [
        'uv', 'run', 'rogii-well-gate',
        '--config', 'configs/experiment/stage8b_conditional_well_gate.yaml',
        '--cutback-run', str(cutback_run),
        '--artifact-dir', str(artifact_dir),
        '--run-id', RUN_ID,
    ]
    log_path = artifact_dir / f'{RUN_ID}_driver.log'
    with log_path.open('w', encoding='utf-8') as log_handle:
        process = subprocess.Popen(
            command, cwd=repo_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end='')
            log_handle.write(line); log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        tail = log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-80:]
        print('\\n'.join(tail))
        raise RuntimeError(f'Stage 8B failed with exit code {return_code}. Full log: {log_path}')
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
    'fold_deltas': gate['fold_deltas'],
    'gates': gate['gates'],
    'spatial_delta': gate['spatial']['pooled_rmse_delta'],
    'spatial_fold_deltas': gate['spatial']['fold_deltas'],
    'inference_threshold': gate['inference_threshold'],
    'standard_selections': gate['standard_selections'],
    'spatial_selections': gate['spatial_selections'],
    'threshold_report': gate['threshold_report'],
}
"""),
    markdown("""## Decision

Only `promoted: True` with a non-null `inference_threshold` authorizes Kaggle integration. Otherwise the physics/gating line is closed and development moves to a new residual sequence model.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "stage8b": {"submission": False, "standalone": True, "target_in_features": False},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
