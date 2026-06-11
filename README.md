# DDPM for Anomaly Detection in Mammography Images

This project implements a diffusion-based anomaly detection pipeline for mammography images.

The main idea is to train a denoising diffusion model on healthy images and then use reconstruction differences to detect suspicious regions. During evaluation, the model produces anomaly maps and anomaly scores that can be compared with available segmentation masks.

## Project Structure

```text
ddpm-for-anomaly-detection/
├── configs/
│   └── inbreast_debug.yaml
├── scripts/
│   ├── prepare_dataset.py
│   └── check_dataset.py
├── src/
│   ├── data/
│   │   └── inbreast_dataset.py
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── results/
│   ├── initial_eval/
│   └── balanced_eval/
└── docs/
```

## Dataset Preparation

The raw dataset is first converted into a simplified folder structure used by the training and evaluation pipeline:

```text
datasets/
└── INBreast/
    ├── train/
    │   └── healthy/
    └── test/
        ├── healthy/
        ├── mass/
        └── masks/
```

The preparation script converts DICOM images to grayscale PNG files, resizes them to 256×256 pixels and prepares binary masks for mass cases.

```bash
python scripts/prepare_dataset.py --raw-root dataset_raw/INBreast --output-root datasets/INBreast --overwrite
```

The prepared dataset can be checked with:

```bash
python scripts/check_dataset.py --data-root datasets/INBreast
```

## Method Overview

The model follows a reconstruction-based anomaly detection approach:

1. Train the diffusion model using healthy mammography images.
2. During evaluation, reconstruct test images.
3. Compute anomaly maps from reconstruction differences.
4. Compare anomaly maps and anomaly scores with available labels and masks.

The assumption is that a model trained on healthy images should reconstruct normal tissue more reliably than abnormal regions.

## Current Results

The current repository contains lightweight evaluation artifacts in the `results/` folder.

Two small evaluation settings are included:

* `initial_eval`: first mini-evaluation on a small unbalanced subset
* `balanced_eval`: mini-evaluation on a balanced subset

The initial mini-evaluation showed a sample-level anomaly detection signal, while pixel-level localization remained weak. This is expected because the available experiment was limited by computational resources and used a very short training run.

Large files such as checkpoints, optimizer states and model weights are intentionally excluded from the repository.

## Notes

This repository contains the cleaned project version prepared for presentation and defense. The focus is on a clear and explainable pipeline rather than on large-scale training.
