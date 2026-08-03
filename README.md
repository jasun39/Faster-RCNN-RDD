# Faster-RCNN-RDD

This repository provides a PyTorch implementation of Faster R-CNN for the Road Damage Detection (RDD) dataset. It supports object detection for road surface damage classification using ResNet-18 and ResNet-50-FPN backbones.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)

## Overview

The project detects and classifies road defects into specific categories. It includes scripts to train models, run evaluation metrics using COCO tools, generate detection visual outputs, and analyze prediction results.

## Project Structure

```text
Faster-RCNN-RDD/
├── config/
│   ├── dataset_dir_structure.txt
│   ├── environment.yaml
│   ├── environment_training.yaml
│   ├── rdd.yaml
│   └── requirements.txt
├── dataset/
│   └── README.md
├── models/
│   ├── __init__.py
│   ├── create_fasterrcnn_model.py
│   ├── fasterrcnn_resnet18.py
│   ├── fasterrcnn_resnet50_fpn.py
│   └── model_summary.py
├── torch_utils/
│   ├── coco_eval.py
│   ├── coco_utils.py
│   ├── engine.py
│   └── utils.py
├── utils/
│   ├── general.py
│   └── logging.py
├── analyze_results.py
├── datasets.py
├── test_model.py
├── train.py
└── visualize.py
```

## Requirements

Install dependencies using Conda or pip.

### Option 1: Conda Environment

Create the environment from the provided YAML file:

```bash
conda env create -f config/environment.yaml
conda activate rdd-env
```

### Option 2: Pip Requirements

Install packages directly with pip:

```bash
pip install -r config/requirements.txt
```

## Dataset Preparation

1. Read `dataset/README.md` for download instructions.
2. Format the directory tree as specified in `config/dataset_dir_structure.txt`.
3. Verify that the dataset configuration file `config/rdd.yaml` contains correct paths and class labels.

## Training

Run `train.py` to start model training.

```bash
python train.py --config config/rdd.yaml
```

Main parameters:
- `--config`: Path to dataset and training configuration file.
- `--model`: Select model architecture (`resnet50_fpn` or `resnet18`).
- `--epochs`: Number of training epochs.
- `--batch-size`: Number of samples per batch.

## Evaluation

Evaluate a trained model using COCO metrics with `test_model.py`.

```bash
python test_model.py --config config/rdd.yaml --weights path/to/weights.pth
```

Run `analyze_results.py` to aggregate evaluation output and plot performance statistics.

```bash
python analyze_results.py --input-dir outputs/
```

## Visualization

Generate visual predictions on sample images using `visualize.py`.

```bash
python visualize.py --config config/rdd.yaml --weights path/to/weights.pth --input path/to/images/
```

This script saves images with bounding boxes and class labels in the output directory.
