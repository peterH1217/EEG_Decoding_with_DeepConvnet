#Final_DS_Project

# Neuro Deep Learning: Motor Imagery Classification

Final project for the Advanced Python / Deep Learning Workshop.
This repository implements the Deep ConvNet architecture (Schirrmeister et al., 2017) on the BCI Competition IV-2a dataset.

## Team
* **Helen:** Data Engineering & Preprocessing Pipeline
* **Peter:** Trial Segmentation & Cropped Training Loop
* **Margaret:** Deep ConvNet Architecture & Model Definition

## Quick Start

### 1. Install Dependencies
Run this command to install all required libraries:
```bash
pip install mne moabb torch scikit-learn pandas matplotlib seaborn

```

### 2. Run the Data Pipeline
To download the dataset (automatically handled via MOABB), filter, normalize, and visualize the data, run:
```bash
python -m src.test_run
```

### 3. Project structure:
**src/data_loader.py:** **Dynamic Loading:** Automatically switches between BCI 2a (22 channels) and HGD (128 channels). **Preprocessing:** Applies 4Hz High-pass filtering (preserving Gamma band up to 125Hz) and channel-wise Z-score normalization. **Segmentation:** Implements the Cropped Training logic (sliding windows) resulting in ~625 crops per trial.

**src/visualization.py:** Generates PSD (Power Spectral Density) plots to verify filtering. Plots Raw EEG Traces to verify normalization and signal quality. Saves outputs dynamically to results/ (e.g., psd_plot_Schirrmeister2017.png).

**src/config.py:** Central configuration (Sampling Rate, Channel Names, Filter settings) that can be adjusted to one's liking. 

**src/test_run.py:** The main script to run the pipeline and generate results.