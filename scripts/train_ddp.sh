#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train_multigpu.yaml}"
NUM_PROCS="${NUM_PROCS:-4}"

torchrun --nproc_per_node="${NUM_PROCS}" -m train.train --config "${CONFIG_PATH}"
