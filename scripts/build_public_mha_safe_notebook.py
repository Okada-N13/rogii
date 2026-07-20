from __future__ import annotations

import argparse
import json
from pathlib import Path


REMOVE_FINGERPRINTS = (
    "Guarded contact override v2",
    "Version 2 artifact audit",
    "Global bias correction: MEASURED on the hidden public test",
    "GATED OFF 2026-07-03: smoother SHELVED",
    "DIAGNOSTIC: exploit dependence",
    "Legal heel-cal predictor v2",
    "Smoother SALVAGE sweep",
    "Smoother-vs-STACK OOF blend test",
    "LB PROBE v4",
)


ATTRIBUTION_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# ROGII public MHA — safe submission build\n",
        "\n",
        "Derived with attribution from Can Qiang's public Kaggle notebook "
        "[`rogii-det-mha140sep4`](https://www.kaggle.com/code/canqiang/rogii-det-mha140sep4).\n",
        "\n",
        "This build retains the 128-seed likelihood PF, learned blend, visible-prefix calibration, "
        "and direction-free midpoint hedge. It removes leaderboard probes, probe-decoded global bias, "
        "same-well contact target transfer, and read-only experimental cells. The final cell validates "
        "one unambiguous `submission.csv`.\n",
    ],
}


FINAL_AUDIT_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Final safe artifact: validate submission.csv and remove ambiguous submission-shaped copies.\n",
        "import glob as _safe_glob, hashlib as _safe_hashlib, json as _safe_json\n",
        "from pathlib import Path as _SafePath\n",
        "import numpy as _safe_np, pandas as _safe_pd\n",
        "_SAFE_W = _SafePath('/kaggle/working') if _SafePath('/kaggle/working').exists() else _SafePath('.')\n",
        "_SAFE_DATA = _SafePath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')\n",
        "if not (_SAFE_DATA / 'sample_submission.csv').is_file():\n",
        "    _safe_hits = _safe_glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True)\n",
        "    if not _safe_hits: raise RuntimeError('sample_submission.csv not found')\n",
        "    _SAFE_DATA = _SafePath(_safe_hits[0]).parent\n",
        "_safe_path = _SAFE_W / 'submission.csv'\n",
        "_safe_sub = _safe_pd.read_csv(_safe_path)[['id', 'tvt']]\n",
        "_safe_sample = _safe_pd.read_csv(_SAFE_DATA / 'sample_submission.csv')\n",
        "if len(_safe_sub) != len(_safe_sample): raise RuntimeError('submission row-count mismatch')\n",
        "if not _safe_sub['id'].astype(str).equals(_safe_sample['id'].astype(str)): raise RuntimeError('submission id/order mismatch')\n",
        "if not _safe_np.isfinite(_safe_sub['tvt'].to_numpy(dtype=float)).all(): raise RuntimeError('non-finite submission tvt')\n",
        "_safe_sub.to_csv(_safe_path, index=False)\n",
        "_removed = []\n",
        "for _safe_file in sorted(_SAFE_W.glob('*.csv')):\n",
        "    if _safe_file == _safe_path: continue\n",
        "    try:\n",
        "        _safe_head = _safe_pd.read_csv(_safe_file, nrows=2)\n",
        "        if {'id', 'tvt'}.issubset(_safe_head.columns):\n",
        "            _safe_file.unlink(); _removed.append(_safe_file.name)\n",
        "    except Exception: pass\n",
        "_safe_sha = _safe_hashlib.sha256(_safe_path.read_bytes()).hexdigest()\n",
        "_safe_audit = {'rows': int(len(_safe_sub)), 'id_order_matches_sample': True, 'finite_tvt': True, "
        "'submission_sha256': _safe_sha, 'removed_ambiguous_csvs': _removed, "
        "'probe_bias_removed': True, 'lb_probe_removed': True, 'contact_target_transfer_removed': True}\n",
        "(_SAFE_W / 'stage6_public_mha_safe_audit.json').write_text(_safe_json.dumps(_safe_audit, indent=2), encoding='utf-8')\n",
        "print(_safe_audit)\n",
        "_safe_sub.head()\n",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-owner")
    parser.add_argument("--kernel-metadata", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    retained = []
    removed = []
    for index, cell in enumerate(payload["cells"]):
        source = "".join(cell.get("source", []))
        fingerprint = next((value for value in REMOVE_FINGERPRINTS if value in source), None)
        if fingerprint is not None:
            removed.append({"index": index, "fingerprint": fingerprint})
            continue
        if index == 0:
            cell = dict(cell)
            cell["source"] = [
                "import os\n",
                "os.environ['ROGII_GOLD_PROFILE'] = 'conservative'\n",
                "os.environ['ROGII_GOLD_PREFIX_CAL'] = '1'\n",
            ]
        retained.append(cell)

    if not any("DELTA midhedge" in "".join(cell.get("source", [])) for cell in retained):
        raise RuntimeError("MHA cell was not found in source notebook")
    combined = "\n".join("".join(cell.get("source", [])) for cell in retained)
    forbidden = ("ROGII_PROBE", "probe canary", "_BC_SHIFT", "LB PROBE v4")
    found = [token for token in forbidden if token in combined]
    if found:
        raise RuntimeError(f"Forbidden probe/bias tokens remain: {found}")

    payload["cells"] = [ATTRIBUTION_CELL, *retained, FINAL_AUDIT_CELL]
    payload.setdefault("metadata", {})["stage6_safe_build"] = {
        "source": "canqiang/rogii-det-mha140sep4",
        "removed_cells": removed,
        "probe_bias_removed": True,
        "contact_target_transfer_removed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output} with {len(payload['cells'])} cells; removed {len(removed)} cells")
    if args.kernel_metadata is not None:
        if not args.kernel_owner:
            raise ValueError("--kernel-owner is required with --kernel-metadata")
        metadata = {
            "id": f"{args.kernel_owner}/rogii-public-mha-safe-v1",
            "title": "ROGII public MHA safe v1",
            "code_file": args.output.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": [
                "phongnguyn23021656/koolbox-offline",
                "fleongg/rogii-claude-models-pub",
                "ravaghi/wellbore-geology-prediction-artifacts",
            ],
            "kernel_sources": [],
            "competition_sources": ["rogii-wellbore-geology-prediction"],
            "model_sources": [],
        }
        args.kernel_metadata.parent.mkdir(parents=True, exist_ok=True)
        args.kernel_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"wrote private-kernel metadata {args.kernel_metadata}")


if __name__ == "__main__":
    main()
