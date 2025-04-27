# Anomaly Detector Project Report

## 1. Introduction

This project implements an anomaly detector using a combined CNN+LSTM architecture. The detector is designed to identify abnormal pulses in beam configuration and DCM (Device Configuration Management) data. The framework is modular, with separate components for data parsing, data preparation, model building, and evaluation. Both training and testing workflows have been implemented, and evaluation metrics are exported as PDF reports.

## 2. Data Sources and Preprocessing

### Data Sources

- **BPM Data (Beam Parameter Data):**  
  Beam configuration is loaded from a CSV file. The file contains columns representing various beam parameters (e.g., Tuner Position, Voltage Settings, etc.). Timestamps are converted to datetime objects, and additional columns are added/renamed using the `BPMDataConfig` class (located in `data_parser/bpm_parser.py`).

- **DCM Data:**  
  DCM data is collected by scanning directories for files labeled as "normal" or "anomal" using the `DCMDatConfig` class (located in `data_parser/dcm_config.py`). The data is processed to extract traces and relevant metadata.

### Preprocessing Steps

- **Merging:**  
  BPM and DCM data are merged based on timestamps using a merge-as-of approach.

- **Cleaning:**  
  The merged dataset is cleaned using the `DataPreprocessor` class (located in `data_preparation/data_preprocessor.py`) to remove NaN values, duplicates, and outliers.

- **Feature Extraction:**  
  - **Traces:** Time-series traces are extracted and normalized.  
  - **Scalar Features:** Numeric features (excluding the anomaly flag) are normalized.  
  The two feature sets are concatenated to form the final feature matrix.

- **Handling Missing Values:**  
  Before applying SMOTE, the code checks for missing values in the combined features and drops any rows containing NaNs.

- **Oversampling:**  
  SMOTE is applied to balance the classes with a target minority class ratio of 0.2.

- **Randomization:**  
  The merged dataset is randomized (shuffled) to ensure a randomized sample for evaluation.

## 3. Model Architecture

The anomaly detector model is a hybrid CNN+LSTM network:

- **CNN Layers:** Extract local temporal features from the time-series (trace) input.
- **LSTM Layer:** Captures longer-term temporal dependencies.
- **Scalar Input:** Additional scalar features are concatenated with the CNN+LSTM output.
- **Dense Layers:** Fully connected layers process the combined features, culminating in a sigmoid output for binary classification.

A custom focal loss function is used during training to mitigate class imbalance.

## 4. Training and Evaluation Methodology

### Training Workflow

1. **Data Preparation:**  
   - Load BPM data using `BPMDataConfig`.  
   - Load DCM data using `DCMDatConfig`.  
   - Merge, clean, and preprocess the data (including feature extraction and normalization).  
   - Apply SMOTE to balance the dataset.
2. **Model Training:**  
   - The model is built using the `AnomalyModel` class.  
   - The model is trained with early stopping and checkpoint callbacks.  
   - The best model weights are saved (e.g., as `best_model.keras`).
3. **Evaluation:**  
   - After training, the model is evaluated on a hold-out validation set.  
   - Evaluation metrics and plots (Confusion Matrix, ROC Curve, Precision-Recall Curve) are generated and exported as a PDF (e.g., `train_metrics_report.pdf`).

### Testing Workflow

1. **Random Data Selection:**  
   - The merged dataset is shuffled to randomize the sample selection.
2. **Preprocessing and Feature Extraction:**  
   - The same cleaning and feature extraction steps as in training are applied.
3. **Model Evaluation:**  
   - A pre-trained model (loaded from `final_anomaly_detector.keras`) is evaluated on the test set.
   - Evaluation metrics and plots are exported as a PDF report (e.g., `test_metrics_report.pdf`).



## 6. Discussion

- **Precision & Specificity:**  
Both training and testing metrics show perfect precision and specificity (1.0000), indicating that the model makes very few (or no) false positive predictions.

- **Recall:**  
The recall (TPR) is moderate (0.4545 in training and 0.2987 in testing), meaning the model misses a number of true anomalies.

- **Overall Performance:**  
The F1 score and MCC indicate a balanced performance; however, the relatively low recall suggests further improvements in anomaly detection (e.g., model tuning, feature engineering, or data augmentation) may be needed.

## 7. Conclusion

This project presents a comprehensive pipeline for anomaly detection in beam configuration and DCM data using a CNN+LSTM model. The modular design enables flexible adjustments in data preprocessing, model training, and evaluation. Although the model achieves excellent precision and specificity, the moderate recall highlights the need for further refinement to improve sensitivity toward anomalies. Future work may involve exploring advanced imputation strategies, alternative oversampling techniques, or different model architectures.



