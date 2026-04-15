#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/inference.yaml}"

python -m infer.augment_image --config "${CONFIG_PATH}" --mode translator
python -m infer.apply_params --config "${CONFIG_PATH}"
