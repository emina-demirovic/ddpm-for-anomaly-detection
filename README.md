# DDPM for Anomaly Detection in Mammography Images

This repository contains a cleaned deep learning project for diffusion-based anomaly detection in mammography images.

The project investigates a reconstruction-based approach in which a denoising diffusion model is trained to reconstruct healthy breast tissue. During evaluation, test images are reconstructed and anomaly maps are obtained from the difference between the original and reconstructed image. The main goal of the repository is to present a clear and reproducible pipeline, including dataset preparation, dataset validation, data loading, experiment configuration, Colab workflow and selected initial results.

The project is intended as an academic deep learning project and should not be interpreted as a clinically validated diagnostic system.

---

## Project Overview

The anomaly detection pipeline follows the idea that a model trained on healthy images should reconstruct normal tissue more reliably than abnormal regions. If a suspicious region is not reconstructed well, the reconstruction error can be used as an anomaly signal.

The implemented workflow includes:

1. preprocessing raw mammography data into a simplified folder structure;
2. validating the prepared dataset;
3. loading healthy images for training;
4. loading both healthy and mass cases for evaluation;
5. generating reconstruction-based anomaly maps;
6. comparing anomaly maps with available segmentation masks;
7. saving lightweight evaluation artifacts for presentation and discussion.

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
│   ├── initial_eval/
│   └── balanced_eval/
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Main components

* `scripts/prepare_dataset.py`
  Converts raw DICOM images and annotation files into a simplified image folder structure used by the project.

* `scripts/check_dataset.py`
  Validates the prepared dataset structure, image sizes, mask matching and basic mask correctness.

* `src/data/inbreast_dataset.py`
  Contains the PyTorch dataset and dataloader logic. Training uses healthy images, while evaluation uses both healthy and mass cases. Mass cases have corresponding binary masks, while healthy cases are assigned empty masks.

* `configs/inbreast_debug.yaml`
  Contains the experiment configuration used for the reduced debug setup.

* `src/training_and_evaluation/ddpm_colab_workflow.ipynb`
  Contains the Colab-based workflow used for setup, training, checkpoint loading, evaluation and visualization.

* `results/`
  Contains selected lightweight evaluation artifacts, such as CSV files, text summaries and visualization images.

---

## Dataset Preparation

The raw data are not included in this repository.

After obtaining the raw dataset files locally, the preprocessing script converts the data into the following structure:

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

Run dataset preparation with:

```bash
python scripts/prepare_dataset.py \
    --raw-root dataset_raw/INBreast \
    --output-root datasets/INBreast \
    --image-size 256 \
    --overwrite
```

The script performs the following steps:

* reads DICOM images;
* handles grayscale image conversion;
* normalizes image intensities;
* resizes images to 256 × 256 pixels;
* prepares binary masks for mass cases;
* saves images and masks as PNG files;
* creates the folder structure expected by the dataloader.

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

This step is useful before running training or evaluation, because many errors in medical image experiments come from incorrect dataset organization or mismatched masks.

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

Install dependencies:

```bash
pip install -r requirements.txt
```

For training and evaluation, a GPU environment is strongly recommended. The main experiment workflow was prepared for Google Colab.

---

## Experiment Configuration

The provided configuration file is:

```text
configs/inbreast_debug.yaml
```

It defines the main dataset, dataloader, model, diffusion, training, evaluation and results settings.

The included setup is intentionally small and resource-aware. It was used to verify that the full pipeline works under limited GPU availability. It is not intended to represent a fully optimized training configuration.

---

## Colab Workflow

The main training and evaluation workflow is provided in:

```text
src/training_and_evaluation/ddpm_colab_workflow.ipynb
```

The notebook is used for:

* setting up the environment;
* connecting to the dataset location;
* running the configured experiment;
* loading checkpoints;
* performing evaluation;
* saving anomaly maps, scores and visualizations.

The notebook is included as part of the project workflow because the available local machine was not suitable for full training. GPU-based execution was therefore handled through Colab.

---

## Results

The repository contains selected lightweight result files in:

```text
results/
```

Two evaluation folders are included:

* `initial_eval/`
  First mini-evaluation on a small unbalanced subset.

* `balanced_eval/`
  Mini-evaluation on a balanced subset.

The initial mini-evaluation showed that a sample-level anomaly signal exists, but pixel-level localization remained weak. This is expected because the available run was very short and constrained by limited GPU resources.

For the initial mini-evaluation, the following values were obtained:

| Metric                        |  Value |
| ----------------------------- | -----: |
| Sample-wise average precision | 0.8802 |
| Sample-wise AUROC             | 0.7143 |

These results should be interpreted carefully. They show that the pipeline is functional and that reconstruction differences can provide an initial anomaly signal, but they do not demonstrate a fully optimized lesion localization model.

---

## Limitations

The main limitations of the current project version are:

* training was limited by available GPU resources;
* only a short debug-level training run was completed;
* evaluation was performed on small subsets;
* pixel-level localization is still weak;
* reconstruction errors can also appear near breast boundaries or other high-contrast regions;
* further training and systematic hyperparameter tuning would be required for stronger conclusions.

The current repository therefore focuses on a clean, explainable and reproducible project pipeline rather than on final clinical-level performance.

---

## Large Files

Large files are intentionally excluded from the repository, including:

* model checkpoints;
* optimizer states;
* binary model weights;
* raw dataset files;
* large intermediate experiment folders.

Only lightweight files useful for presentation, inspection and discussion are included.

---

## Project Status

The current version represents a cleaned project submission prepared for presentation and defense. It demonstrates the complete workflow from dataset preparation to initial reconstruction-based anomaly evaluation.

Natural next steps would include longer training, more systematic evaluation, comparison with simpler reconstruction-based baselines and further tuning of the anomaly map post-processing.
