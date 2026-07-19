from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rogii.artifacts import write_json
from rogii.evaluation.metrics import evaluate_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved ROGII OOF predictions")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.predictions.suffix.lower() == ".parquet":
        predictions = pd.read_parquet(args.predictions)
    else:
        predictions = pd.read_csv(args.predictions)
    metrics, wells = evaluate_predictions(predictions)
    output_dir = args.output_dir or args.predictions.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", metrics)
    wells.to_parquet(output_dir / "per_well_metrics.parquet", index=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

