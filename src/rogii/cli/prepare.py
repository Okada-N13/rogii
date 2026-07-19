from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rogii.artifacts import environment_report, write_json
from rogii.config import resolve_artifact_dir, resolve_data_dir
from rogii.data.folds import assert_group_isolation, make_group_folds
from rogii.data.loading import dataset_fingerprint, discover_horizontal_wells, load_horizontal_well
from rogii.data.schema import validate_horizontal_well


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate ROGII data and create fixed well folds")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-wells", type=int)
    parser.add_argument("--skip-fingerprint", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_dir = (args.data_dir or resolve_data_dir()).resolve()
    artifact_root = (args.artifact_dir or resolve_artifact_dir()).resolve()
    output_dir = artifact_root / "prepared"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, list[dict[str, object]]] = {}
    all_paths: list[Path] = []
    for split in ("train", "test"):
        paths = discover_horizontal_wells(data_dir, split)
        if args.limit_wells is not None:
            paths = paths[: args.limit_wells]
        records: list[dict[str, object]] = []
        for index, path in enumerate(paths, start=1):
            frame = load_horizontal_well(path)
            stats = validate_horizontal_well(frame, split=split)
            records.append(stats.to_dict())
            if index % 100 == 0:
                print(f"validated {index}/{len(paths)} {split} wells", flush=True)
        all_stats[split] = records
        all_paths.extend(paths)
        pd.DataFrame.from_records(records).to_parquet(output_dir / f"{split}_well_stats.parquet", index=False)

    train_stats = pd.DataFrame.from_records(all_stats["train"])
    folds = make_group_folds(train_stats, n_splits=args.n_splits, seed=args.seed)
    assert_group_isolation(folds)
    folds.to_parquet(output_dir / "folds.parquet", index=False)

    report = {
        "data_dir": str(data_dir),
        "n_train_wells": len(all_stats["train"]),
        "n_test_wells": len(all_stats["test"]),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "dataset_fingerprint": None if args.skip_fingerprint else dataset_fingerprint(all_paths),
        "environment": environment_report(),
    }
    write_json(output_dir / "data_report.json", report)
    print(f"prepared artifacts: {output_dir}")


if __name__ == "__main__":
    main()

