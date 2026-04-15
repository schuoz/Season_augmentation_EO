# Seasonal Remote-Sensing Augmentation Repo

PyTorch-based training and inference pipeline for learning seasonal effects from paired remote-sensing imagery. The project trains:

- an image-to-image seasonal translator, and
- a parameter predictor head that outputs reusable augmentation controls.

## Data contract

Prepare paired metadata CSV with columns:

- `source_path`
- `target_path`
- `source_season`
- `target_season`
- `location_id`
- `date_source`
- `date_target`

Paths should point to GeoTIFF images.

## Repository layout

- `configs/`: train/inference configs
- `src/data/`: dataset, raster IO, distributed sampler helpers
- `src/models/`: generator, discriminator, parameter head, losses
- `src/train/`: DDP train loop, engine, checkpointing
- `src/infer/`: augmentation and export scripts
- `scripts/`: helper shell/python scripts
- `tests/`: smoke tests

## Quickstart

Install:

```bash
pip install -e .[dev]
```

Prepare metadata:

```bash
python scripts/prepare_pairs.py --input_dir /path/to/data --output_csv /path/to/pairs.csv
```

Train (DDP):

```bash
bash scripts/train_ddp.sh configs/train_multigpu.yaml
```

Single-process debug:

```bash
python -m train.train --config configs/train_baseline.yaml
```

Run translator inference:

```bash
python -m infer.augment_image --config configs/inference.yaml --mode translator
```

Run parameter-only augmentation:

```bash
python -m infer.apply_params --config configs/inference.yaml
```

Export weights package:

```bash
python -m infer.export_weights --checkpoint_dir outputs/checkpoints --export_dir outputs/export
```
