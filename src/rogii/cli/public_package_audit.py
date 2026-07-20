from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from rogii.artifacts import write_json


KEYWORDS = ("oof", "fold", "pred", "blend", "manifest", "metric", "feature", "config")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit public model packages for honest OOF assets")
    parser.add_argument("--package", action="append", required=True, help="label=/path/to/package")
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _ordered_hash(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _json_summary(value: Any, depth: int = 0) -> Any:
    if depth >= 3:
        return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        return {str(key): _json_summary(item, depth + 1) for key, item in list(value.items())[:100]}
    if isinstance(value, list):
        if len(value) <= 30:
            return [_json_summary(item, depth + 1) for item in value]
        return {"type": "list", "length": len(value), "head": [_json_summary(x, depth + 1) for x in value[:10]]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _tabular_summary(path: Path, base_rows: int, base_id_hash: str) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        metadata = pq.ParquetFile(path)
        columns = metadata.schema.names
        rows = metadata.metadata.num_rows
        sample = metadata.read_row_group(
            0, columns=columns[: min(12, len(columns))]
        ).slice(0, 3).to_pandas()
    else:
        sample = pd.read_csv(path, nrows=3)
        columns = sample.columns.tolist()
        rows = None
    result: dict[str, Any] = {
        "columns": columns,
        "rows": rows,
        "sample": _json_summary(
            sample.astype(object).where(pd.notna(sample), None).to_dict(orient="records")
        ),
        "row_count_matches_base": rows == base_rows if rows is not None else None,
    }
    id_columns = [name for name in columns if name.lower() in {"id", "row_id"}]
    interesting = any(word in path.name.lower() for word in ("oof", "pred", "blend"))
    if id_columns and interesting:
        id_column = id_columns[0]
        if suffix == ".parquet":
            ids = pd.read_parquet(path, columns=[id_column])[id_column]
        else:
            ids = pd.read_csv(path, usecols=[id_column])[id_column]
            rows = len(ids)
            result["rows"] = rows
            result["row_count_matches_base"] = rows == base_rows
        result["id_column"] = id_column
        result["ordered_id_sha256"] = _ordered_hash(ids)
        result["id_order_matches_base"] = result["ordered_id_sha256"] == base_id_hash
    numeric_candidates = [
        name for name in columns
        if any(word in name.lower() for word in ("pred", "oof", "blend", "tvt", "target"))
    ]
    result["prediction_candidate_columns"] = numeric_candidates
    return result


def _array_summary(path: Path, base_rows: int) -> dict[str, Any]:
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "first_dimension_matches_base": bool(array.ndim and array.shape[0] == base_rows),
        }
    archive = np.load(path, allow_pickle=False)
    arrays = {
        name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)}
        for name in archive.files
    }
    return {"arrays": arrays}


def audit_package(label: str, root: Path, base_rows: int, base_id_hash: str) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    inventory = [
        {"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size}
        for path in files
    ]
    candidates: list[dict[str, Any]] = []
    json_documents: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    for path in files:
        relative = str(path.relative_to(root))
        lower = relative.lower()
        if path.suffix.lower() == ".json":
            try:
                json_documents[relative] = _json_summary(json.loads(path.read_text(encoding="utf-8")))
            except Exception as error:
                errors.append({"path": relative, "error": repr(error)})
        if not any(keyword in lower for keyword in KEYWORDS):
            continue
        row: dict[str, Any] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix.lower(),
        }
        try:
            if path.suffix.lower() in {".csv", ".parquet"}:
                row["content"] = _tabular_summary(path, base_rows, base_id_hash)
            elif path.suffix.lower() in {".npy", ".npz"}:
                row["content"] = _array_summary(path, base_rows)
        except Exception as error:
            row["inspection_error"] = repr(error)
        candidates.append(row)
    return {
        "label": label,
        "root": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "inventory": inventory,
        "candidate_files": candidates,
        "json_documents": json_documents,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    base = pd.read_parquet(args.base_run / "base_oof.parquet", columns=["id"])
    base_hash = _ordered_hash(base["id"])
    packages: list[dict[str, Any]] = []
    for value in args.package:
        if "=" not in value:
            raise ValueError(f"Expected label=path, got {value!r}")
        label, path = value.split("=", 1)
        packages.append(audit_package(label, Path(path), len(base), base_hash))
    report = {
        "schema_version": "rogii_public_package_audit_v1",
        "base_rows": len(base),
        "base_ordered_id_sha256": base_hash,
        "packages": packages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    compact = {
        "base_rows": report["base_rows"],
        "packages": [
            {
                "label": package["label"],
                "file_count": package["file_count"],
                "total_gib": package["total_bytes"] / (1024**3),
                "candidate_files": package["candidate_files"],
                "json_documents": package["json_documents"],
                "errors": package["errors"],
            }
            for package in packages
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"full report: {args.output}")


if __name__ == "__main__":
    main()
