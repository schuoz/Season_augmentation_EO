# Seasonal Remote-Sensing Augmentation

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)

A PyTorch-based training and inference pipeline for learning seasonal effects from paired remote-sensing imagery. 

Instead of just generating synthetic images, this project trains a network to predict **reusable, interpretable augmentation controls** (like channel-wise gain and bias). This allows you to apply realistic seasonal changes to your satellite imagery on-the-fly, serving as a powerful and highly efficient data augmentation strategy for downstream Remote Sensing ML tasks.

## Why this approach?

Traditional Image-to-Image translation models (like CycleGAN or Pix2Pix) generate physically plausible images, but running a deep generator during your data loading pipeline is prohibitively slow. 

This repository solves that by co-training two outputs:
1. **An Image-to-Image Seasonal Translator**: A GAN-based generator that learns to translate an image from a source season to a target season.
2. **An Augmentation Parameter Head**: Predicts a set of simple, fast-to-apply transformation parameters (gain and bias for each channel) that approximate the complex translation.

During inference, you can use the lightweight **parameter-only** augmentation, applying a fast affine transformation directly to your images instead of passing them through the entire heavy generator network.

## 🗂️ Repository Structure

```text
├── configs/             # YAML configuration files for training and inference
├── scripts/             # Shell scripts and utilities (data prep, DDP launch)
├── src/
│   ├── data/            # GeoTIFF loading, dataset sampling, filtering
│   ├── models/          # Generator, PatchDiscriminator, and Loss functions
│   ├── train/           # Distributed Data Parallel (DDP) engine and loops
│   └── infer/           # Inference scripts (translator and parameter application)
└── tests/               # Smoke tests for data and models
```

## 🛠️ Installation

Requirements: Python >= 3.10

```bash
git clone https://github.com/your-username/seasonal-rs-augmentation.git
cd seasonal-rs-augmentation
pip install -e .[dev]
```

## 💾 Data Preparation

The dataset expects paired imagery across different seasons (e.g. summer and winter over the same geographic location). You must prepare a CSV file containing the metadata mapping for these pairs. 

At a minimum, the CSV must contain `source_path` and `target_path`. Paths should point to valid multi-band GeoTIFFs.

Run the helper script to auto-generate this CSV from a directory structure:
```bash
python scripts/prepare_pairs.py --input_dir /path/to/data --output_csv data/pairs.csv
```

## 🚀 Quickstart

### 1. Training

**Single GPU (Debug/Baseline):**
```bash
python -m train.train --config configs/train_baseline.yaml
```

**Multi-GPU (Distributed Data Parallel):**
```bash
bash scripts/train_ddp.sh configs/train_multigpu.yaml
```

*Note: Edit the configuration files in `configs/` to tweak batch size, learning rates, band selections, and data paths.*

### 2. Inference & Augmentation

Once trained, you have two ways to apply the seasonal effects to new images:

**Mode A: Full Image-to-Image Translation**  
Runs the image through the entire deep generator. Produces the most accurate seasonal translation, but is relatively resource intensive.
```bash
python -m infer.augment_image --config configs/inference.yaml --mode translator
```

**Mode B: Parameter-Only Augmentation (Recommended for ML pipelines)**  
Predicts the seasonal gain/bias parameters and applies a lightweight affine transformation. Extremely fast and ideal for injecting seasonal diversity into downstream training pipelines on the fly.
```bash
python -m infer.apply_params --config configs/inference.yaml
```

**Export Weights:**  
You can export just the final weight packages for easy deployment and sharing:
```bash
python -m infer.export_weights \
  --checkpoint_dir outputs/checkpoints \
  --export_dir outputs/export
```
