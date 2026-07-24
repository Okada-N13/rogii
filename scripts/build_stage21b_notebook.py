"""Build the standalone Stage 21B disjoint prefix-confidence notebook."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("notebooks/550_run_stage21b_prefix_confidence.ipynb")


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
            "# Stage 21B: disjoint prefix-confidence gate\n\n"
            "Stage 21Aは内部順位で候補を直接選ぶと悪化しました。この確認ではStage 21Aの63 wellsを"
            "候補固有のinner→outer楽観バイアス推定だけに使い、Stage 20A/BとStage 21Aの全wellを"
            "除外した別sampleで評価します。多項式候補は除外し、補正後marginと2 internal cutsの"
            "両方を通った代替候補だけをA130 baseへ10%・8 ft上限で混ぜます。提出は作りません。"
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
            "## 固定artifact\n\nStage 21A結果はpenalty校正にのみ使います。評価sampleはStage 20A/Bと"
            "Stage 21Aのwellをすべて除外します。\n"
        ),
        code(
            "stage16b_run=artifact_dir/'stage16b_testlike_validation_full_v003'\n"
            "stage17a_run=artifact_dir/'stage17_public_replay_full_v002'\n"
            "public_oof_run=artifact_dir/'stage7_public_residual_gate_full_v001'\n"
            "stage20a_run=artifact_dir/'stage20a_top_pf_alignment_full_v001'\n"
            "stage20b_run=artifact_dir/'stage20b_disjoint_confirmation_full_v001'\n"
            "stage21a_run=artifact_dir/'stage21a_prefix_router_full_v001'\n"
            "required=[stage16b_run/'well_assignments.parquet',stage17a_run/'cut_report.parquet',"
            "public_oof_run/'base_oof.parquet',stage20a_run/'cut_features.parquet',"
            "stage20b_run/'cut_features.parquet',stage21a_run/'summary.json',"
            "stage21a_run/'candidate_report.parquet',stage21a_run/'router_cut_report.parquet']\n"
            "for path in required: assert path.is_file(),path\n"
            "print(*required,sep='\\n')\n"
        ),
        markdown(
            "## Disjoint confidence gate\n\n10 cutsごとに進捗を表示します。Stage 21Aと同程度の"
            "screen解像度です。途中失敗時は、このrunだけを安全に削除して最初から再実行します。\n"
        ),
        code(
            "RUN_ID='stage21b_prefix_confidence_full_v001'; run_dir=artifact_dir/RUN_ID\n"
            "if run_dir.exists() and not (run_dir/'summary.json').is_file():\n"
            "    resolved=run_dir.resolve(); expected=(artifact_dir/RUN_ID).resolve()\n"
            "    assert resolved==expected and resolved.parent==artifact_dir.resolve(),resolved\n"
            "    print('Removing incomplete prior run:',resolved); shutil.rmtree(resolved)\n"
            "if not (run_dir/'summary.json').is_file():\n"
            "    run_checked(['uv','run','rogii-prefix-confidence','--config',"
            "'configs/experiment/stage21b_prefix_confidence.yaml','--stage16b-run',str(stage16b_run),"
            "'--stage17a-run',str(stage17a_run),'--public-oof-run',str(public_oof_run),"
            "'--stage21a-run',str(stage21a_run),'--exclude-run',str(stage20a_run),"
            "'--exclude-run',str(stage20b_run),'--data-dir',str(data_dir),"
            "'--artifact-dir',str(artifact_dir),'--run-id',RUN_ID])\n"
            "summary=json.loads((run_dir/'summary.json').read_text())\n"
            "{key:summary[key] for key in ['stage21b_complete','promoted_to_stage21c','calibration_cuts',"
            "'calibration_wells','sample_cuts','sample_wells','excluded_wells','calibration_well_overlap',"
            "'candidate_count','alternative_accepted_cuts','selected_candidate_counts','base_rmse',"
            "'candidate_rmse','rmse_delta','well_p90_delta','bootstrap_95pct','gates','next_step']}\n"
        ),
        code(
            "import pandas as pd\n"
            "display(pd.DataFrame(summary['calibration_penalties']).sort_values('risk_adjusted_penalty'))\n"
            "pd.DataFrame(summary['weight_report']).sort_values('weight')\n"
        ),
        markdown(
            "最後の辞書、calibration penalty表、weight表を共有してください。全gate通過時だけ"
            "高解像度・all-cut確認へ進めます。不通過ならprefix candidate routingを終了します。\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "CPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "stage21b": {
                "submission": False,
                "standalone_setup": True,
                "disjoint_confirmation": True,
                "polynomial_candidates": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
