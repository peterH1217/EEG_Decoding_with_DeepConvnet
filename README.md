# Neuro Deep Learning: DeepConvNet Replication

> **Project Proposal:** Reproduction of Schirrmeister et al. (2017) "Deep learning with convolutional neural networks for EEG decoding and visualization".

## Team
* **Helen:** Data Engineering & Preprocessing Pipeline
* **Peter:** Trial Segmentation & Cropped Training Loop
* **Margaret:** Deep ConvNet Architecture & Model Definition

---

## Project Overview
**Objective:** Reproduce the performance of the **Deep ConvNet** architecture on motor imagery EEG data using the "Cropped Training" strategy.
**Hypothesis:** Using a sliding window (cropped) approach with a specialized CNN can achieve high accuracy on raw EEG data without complex feature engineering.

---

## Project Structure
The project follows a standard `src` package layout:

```text
neuro_deep_learning/
├── src/neuro_deep_learning/   # Main Package
│   ├── config.py              # Central configuration (Paths, Constants)
│   ├── dataset.py             # Preprocessing & Data Loading
│   ├── fetch.py               # Data acquisition (MOABB)
│   ├── cnn.py                 # DeepConvNet Model Architecture
│   ├── train.py               # Training Loop & Validation
│   ├── visualization.py       # Plotting helpers (PSD, Traces)
│   ├── logger.py              # Centralized logging setup
│   └── grand_average_*.py     # Evaluation Scripts
├── tests/                     # Unit Tests
├── results/                   # Generated Artifacts
│   ├── models/                # Saved .pth models
│   ├── figures/               # Confusion matrices & PSD plots
│   └── grand_average/         # Final aggregate results
├── pyproject.toml             # Dependencies & Build Config
└── README.md                  # Project Documentation

```

## Key Parameters & Configuration
Configuration is centralized in `src/neuro_deep_learning/config.py`.

* **Sampling Rate:** 250 Hz (Resampled to match paper).
* **Filtering:** 4.0 Hz High-pass (removes EOG artifacts) to None (preserves Gamma band).
* **Crop Size:** 500 samples (2 seconds).
* **Stride:** 100 samples (Sliding window overlap).
* **Normalization:** Channel-wise Z-score normalization.

---

## Data Description
We use two benchmark Motor Imagery datasets via the **MOABB** library:

1.  **BCI Competition IV-2a (BNCI2014_001):**
    * 22 EEG channels, 9 subjects.
    * 4 classes: Left Hand, Right Hand, Feet, Tongue.
2.  **High Gamma Dataset (Schirrmeister2017):**
    * up to 128 channels, 14 subjects.
    * 4 classes: Left Hand, Right Hand, Feet, Rest.

---

## Usage Instructions

### 1. Installation
Install the package in editable mode with all dependencies:
```bash
pip install -e .

```

2. Run Training
To train the model on a specific subject (e.g., Subject 1):

```bash
python -m neuro_deep_learning.train
```

# Runs the training pipeline defined in src/neuro_deep_learning/train.py
python -m neuro_deep_learning.train
3. Generate Results (Grand Average)
To calculate the mean accuracy across all subjects and generate the final plots:
```
Bash

# For BCI Competition IV-2a
python -m neuro_deep_learning.grand_average_bnci

# For High Gamma Dataset
python -m neuro_deep_learning.grand_average_schirrmeister2017
```
4. Run Tests
Verify the integrity of the data pipeline and model architecture:
```
Bash

pytest
```

**Pipeline Stages**
**Data Import (fetch.py):** Downloads datasets automatically using MOABB.

**Preprocessing (dataset.py):** Resampling to 250Hz.

**Bandpass filtering (4Hz - Inf).**

**Z-score normalization**

**Modeling (cnn.py):** 4-layer DeepConvNet with MaxPolling and ELU activation.

**Analysis (grand_average_*.py):** Aggregates subject accuracies and generates confusion matrices.

**References**
Schirrmeister, R. T., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. Human Brain Mapping.

MOABB: Mother of all BCI Benchmarks (Jayaram & Barachant, 2018).