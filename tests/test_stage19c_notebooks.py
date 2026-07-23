from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage19c_notebooks_are_clean_and_compile() -> None:
    for relative in [
        "notebooks/500_build_stage19c_inference_package.ipynb",
        "notebooks/510_kaggle_top_pf_stage19_trajectory.ipynb",
    ]:
        payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        for cell in payload["cells"]:
            if cell.get("cell_type") == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs") == []
                compile("".join(cell.get("source", [])), relative, "exec")
    kaggle = json.loads(
        (ROOT / "notebooks/510_kaggle_top_pf_stage19_trajectory.ipynb").read_text(encoding="utf-8")
    )
    text = "\n".join("".join(cell.get("source", [])) for cell in kaggle["cells"])
    assert text.count("STAGE19C_TEST_AUDIT=") == 1
    assert "hidden_target_columns_used" in text
    assert kaggle["metadata"]["stage19c_trajectory"]["internet"] is False
