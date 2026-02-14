# N-CMAPSS RUL Prediction and Healthy State Analysis

This project implements a comprehensive data processing and modeling pipeline for Remaining Useful Life (RUL) prediction of aircraft engines using the N-CMAPSS dataset (DS02). A hybrid approach is employed: first, the "Healthy State" of the engine is modeled using a GRU Autoencoder, and then the derived Health Index (HI) is combined with sensor data to predict RUL using a Mamba2-based model.

## Project Pipeline

The data processing and modeling process consists of the following steps:

### 1. Raw Data
*   **File:** `raw data/N-CMAPSS_DS02-006.h5`
*   **Description:** The raw N-CMAPSS DS02 dataset provided by NASA. It contains flight data, sensor readings, and operational conditions.

### 2. Healthy State Preprocessing
*   **File:** `healthy state/preprocess_healthy_state_unified.py`
*   **Function:** Reads data from the raw H5 file.
    *   Separates Training (`DEV`) and Test (`TEST`) sets.
    *   Applies Savitzky-Golay filter to sensor data (`Xs`) for noise reduction.
    *   **Outputs:**
        *   `DS02_healthyStateTrain.parquet`: Contains only healthy state (`hs=1`) data (for Model training).
        *   `DS02_healthyStateInference.parquet`: Contains all data (for Inference).
        *   `DS02_healthyStateInference_TEST.parquet`: Contains Test data.

### 3. Healthy State Modeling
*   **File:** `healthy state/HS_model_GRU.ipynb`
*   **Model:** GRU-based Autoencoder / Reconstructor.
*   **Function:**
    *   Trained only on healthy flight data (`hs=1`).
    *   Aims to learn the healthy behavior of the engine.
    *   Uses 18 features (4 Operational Conditions + 14 Sensors).
    *   The model reconstructs sensor values for a given flight segment.
*   **Output:** Trained model weights (`best_model.pth`) and normalization parameters (`normalization_params.pkl`).

### 4. Health Index Inference
*   **Note:** This step involves reconstructing the entire dataset (healthy + degraded) using the trained GRU model. The difference between actual values and model predictions (Residuals) represents the engine's degradation level (Health Index).
*   **Intermediate Output:** `healthy state/inference_results.parquet` (This file contains calculated residual/error values).

### 5. Visualization
*   **File:** `healthy state/visualization.py`
*   **Function:** Reads `inference_results.parquet` to plot Health Index and sensor error graphs.
*   **Outputs:** Graphs are saved in `health_index_plots` and `average_sensor_plots` directories.

### 6. RUL Preprocessing
*   **File:** `RUL/Mamba2_preprocess_unified.py`
*   **Function:**
    *   Reads RUL information from the raw H5 file and converts it to continuous RUL.
    *   Merges raw data with `inference_results.parquet` (Residuals) from the Health Index step.
    *   Normalizes data and aligns Training/Validation/Test sets.
*   **Output:** `health_index/mamba2_processed.parquet` (Ready-to-use dataset for model training).

### 7. RUL Prediction (Mamba2)
*   **File:** `RUL/mamba2_RUL.ipynb`
*   **Model:** Mamba2 (State Space Model).
*   **Input:** 37 Features (Cycle + 4 Operational Conditions + 14 Raw Sensors + 18 Residuals/Health Index).
*   **Function:**
    *   Uses the prepared `mamba2_processed.parquet` data.
    *   Splits data into Training (Units: 2, 5, 10, 16, 18) and Validation (Unit: 20).
    *   Predicts RUL values.
    *   Displays results and prediction graphs.

## Requirements

Key Python libraries required for the project:
*   `numpy`, `pandas`, `h5py` (Data processing)
*   `torch` (Deep learning models: GRU, Mamba2)
*   `scikit-learn` (Pre-processing, Scaling)
*   `matplotlib` (Visualization)
*   `pyarrow` (Parquet file operations)
*   `scipy` (Signal processing)

## Execution Order

1.  **Preprocessing:** Run `healthy state/preprocess_healthy_state_unified.py`.
2.  **Model Training (HS):** Run the `healthy state/HS_model_GRU.ipynb` notebook.
3.  **Inference:** (If not done within the notebook) Generate `inference_results.parquet` using the GRU model.
4.  **RUL Preparation:** Run `RUL/Mamba2_preprocess_unified.py`.
5.  **RUL Prediction:** Train and test the Mamba2 model in the `RUL/mamba2_RUL.ipynb` notebook.
