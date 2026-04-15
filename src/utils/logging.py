from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def create_writer(output_dir: str) -> SummaryWriter:
    log_dir = Path(output_dir) / "tb_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))
