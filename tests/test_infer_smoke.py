from __future__ import annotations

import json
from pathlib import Path

from infer.export_weights import main as export_main


def test_export_manifest(tmp_path: Path, monkeypatch) -> None:
    ckpt_dir = tmp_path / "ckpt"
    out_dir = tmp_path / "out"
    ckpt_dir.mkdir()
    (ckpt_dir / "generator_best.pt").write_bytes(b"abc")
    (ckpt_dir / "param_head_best.pt").write_bytes(b"def")

    monkeypatch.setattr(
        "sys.argv",
        [
            "export_weights.py",
            "--checkpoint_dir",
            str(ckpt_dir),
            "--export_dir",
            str(out_dir),
        ],
    )
    export_main()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "generator_best.pt" in manifest["files"]
