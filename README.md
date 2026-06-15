# DDPM for Anomaly Detection in Mammography Images

This repository contains a cleaned deep learning project for diffusion-based anomaly detection in mammography images.

The project explores a reconstruction-based anomaly detection approach. A denoising diffusion model is used to reconstruct normal breast tissue, and anomaly maps are obtained from the difference between the original and reconstructed image. The goal of this repository is to present a clear and reproducible project pipeline, including dataset preparation, dataset validation, data loading, experiment configuration, Colab-based execution and selected initial results.

This project was developed as an academic deep learning project. It is not a clinically validated diagnostic system.

---

## Project Overview

The main idea is based on the assumption that a model trained to reconstruct healthy tissue should reconstruct normal image regions more reliably than abnormal regions. When a suspicious region is not reconstructed well, the reconstruction error can be used as an anomaly signal.

The implemented workflow includes:

1. preparing raw mammography data into a simplified folder structure;
2. validating the prepared dataset;
3. loading healthy images for training;
4. loading both healthy and mass cases for evaluation;
5. generating reconstruction-based anomaly maps;
6. comparing anomaly maps with available segmentation masks;
7. saving lightweight evaluation artifacts for inspection and discussion.

Lesion masks are used only for evaluation. They are not used as direct supervision during training.

---

## Repository Structure

```text
ddpm-for-anomaly-detection/
│
├── configs/
│   └── inbreast_debug.yaml
│
├── scripts/
│   ├── prepare_dataset.py
│   └── check_dataset.py
│
├── src/
│   ├── data/
│   │   └── inbreast_dataset.py
│   ├── models/
│   ├── training_and_evaluation/
│   │   └── ddpm_colab_workflow.ipynb
│   └── utils/
│
├── results/
│   ├── README.md
│   └── initial_eval/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Main Components

### `scripts/prepare_dataset.py`

Prepares the raw dataset for the project. The script reads DICOM images, converts them to grayscale PNG files, resizes them to 256 × 256 pixels and prepares binary masks for mass cases.

### `scripts/check_dataset.py`

Checks whether the prepared dataset has the expected structure. It verifies folder existence, image sizes, mask-image matching and empty masks.

### `src/data/inbreast_dataset.py`

Contains the PyTorch dataset and dataloader logic. Training uses healthy images, while evaluation uses both healthy and mass cases. Mass cases have corresponding binary masks, while healthy cases are assigned empty masks.

### `configs/inbreast_debug.yaml`

Contains the configuration used for the reduced debug experiment. The configuration is intentionally small and resource-aware, because the available training time and GPU resources were limited.

### `src/training_and_evaluation/ddpm_colab_workflow.ipynb`

Contains the Colab-based workflow used during the project. It documents setup, dataset preparation, training, checkpoint loading, evaluation and visualization steps.

### `results/`

Contains selected lightweight evaluation artifacts, such as metric summaries, score tables and visualization images. Large binary files are intentionally excluded.

---

## Dataset Preparation

The raw dataset is not included in this repository.

After obtaining the raw files locally, the preparation script creates the following folder structure:

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

Run the preprocessing script with:

```bash
python scripts/prepare_dataset.py \
    --raw-root dataset_raw/INBreast \
    --output-root datasets/INBreast \
    --image-size 256 \
    --overwrite
```

The script performs the following steps:

* reads DICOM images;
* handles grayscale conversion;
* normalizes image intensities;
* resizes images to 256 × 256 pixels;
* prepares binary masks for mass cases;
* saves images and masks as PNG files;
* creates the folder structure expected by the dataloader.

---

### Dataset Access

The INBreast dataset is not distributed with this repository. Access to the original mammography images and annotations should be requested directly from the dataset authors or the institution responsible for its distribution.

After obtaining authorized access, users should provide the dataset through a local path or upload it to their own execution environment. No direct download link or copy of the dataset is included in this project.

When using the dataset, please cite:  
I. C. Moreira, I. Amaral, I. Domingues, A. Cardoso, M. J. Cardoso and J. S. Cardoso, “INbreast: Toward a Full-Field Digital Mammographic Database,” *Academic Radiology*, vol. 19, no. 2, pp. 236–248, 2012.

--- 

## Dataset Validation

After preprocessing, the prepared dataset can be checked with:

```bash
python scripts/check_dataset.py \
    --data-root datasets/INBreast \
    --image-size 256
```

The validation script checks whether:

* all required folders exist;
* images and masks are present;
* images have the expected resolution;
* mass images have corresponding masks;
* there are no unmatched masks;
* masks are not empty.

This step is important because incorrect dataset organization, mismatched masks or invalid image dimensions can easily lead to misleading training or evaluation results.

---

## Installation

A Python virtual environment is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

A GPU environment is strongly recommended for training and evaluation. The main experimental workflow was prepared for Google Colab.

---

## Experiment Configuration

The provided configuration file is:

```text
configs/inbreast_debug.yaml
```

It contains the main dataset, dataloader, model, diffusion, training, evaluation, checkpointing and results settings.

The included setup was used to verify that the complete pipeline can run under limited GPU availability. It should be interpreted as a debug-level configuration, not as a fully optimized training setup.

---

## Colab Workflow

The main workflow is provided in:

```text
src/training_and_evaluation/ddpm_colab_workflow.ipynb
```

The notebook documents the practical execution of the project, including:

* environment setup;
* dataset download and preparation;
* dataset validation;
* training;
* checkpoint inspection;
* evaluation;
* saving and reviewing result artifacts.

The notebook is included because training was performed in a GPU environment rather than on the local machine.

---

## Results

Selected lightweight result files are included in:

```text
results/
```

The current repository contains an initial mini-evaluation. This evaluation showed that a sample-level anomaly signal can be obtained, while pixel-level localization remained weak.

For the initial mini-evaluation, the following values were obtained:

| Metric                        |  Value |
| ----------------------------- | -----: |
| Sample-wise average precision | 0.8802 |
| Sample-wise AUROC             | 0.7143 |
| Optimal DICE over thresholds  | 0.5102 |

These results should be interpreted carefully. They show that the pipeline is functional and that reconstruction differences can provide an initial anomaly signal. They do not demonstrate a fully optimized lesion localization model.

---

## Limitations

The main limitations of the current project version are:

* training was limited by available GPU resources;
* only a short debug-level training run was completed;
* evaluation was performed on a small subset;
* pixel-level localization is still weak;
* reconstruction errors can also appear near breast boundaries and high-contrast regions;
* further training and systematic hyperparameter tuning would be required for stronger conclusions.

The focus of the repository is therefore a clean, explainable and reproducible project pipeline, rather than final clinical-level performance.

---

## Large Files

Large files are intentionally excluded from this repository, including:

* raw dataset files;
* model checkpoints;
* optimizer states;
* binary model weights;
* large intermediate experiment folders.

Only lightweight files useful for presentation, inspection and discussion are included.

---

## Project Status

The current version represents a cleaned project submission prepared for presentation and defense. It demonstrates the complete workflow from dataset preparation to initial reconstruction-based anomaly evaluation.

Natural next steps would include longer training, broader evaluation, comparison with simpler reconstruction-based baselines and further tuning of anomaly map generation.
