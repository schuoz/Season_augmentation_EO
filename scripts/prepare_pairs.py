#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_csv", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows = []
    for location_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        source = location_dir / "source.tif"
        target = location_dir / "target.tif"
        if source.exists() and target.exists():
            rows.append(
                {
                    "source_path": str(source),
                    "target_path": str(target),
                    "source_season": "unknown",
                    "target_season": "unknown",
                    "location_id": location_dir.name,
                    "date_source": "",
                    "date_target": "",
                }
            )
    df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
