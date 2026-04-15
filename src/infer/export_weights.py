from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--export_dir", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    files = ["generator_best.pt", "param_head_best.pt", "model_full.pt"]
    exported = []
    for name in files:
        src = checkpoint_dir / name
        if src.exists():
            shutil.copy2(src, export_dir / name)
            exported.append(name)

    manifest = {"source_checkpoint_dir": str(checkpoint_dir), "files": exported}
    with (export_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
