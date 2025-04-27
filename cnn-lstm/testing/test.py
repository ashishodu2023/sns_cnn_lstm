#!/usr/bin/env python
# coding: utf-8

### Python Packages
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model

### Class Packages
from utils.logger import Logger
from data_preparation.data_loader import TracingBodyDataLoader
from data_preparation.data_transformer import DataMerger, FileGrouping
from data_preparation.data_scaling import DataPreparation
from model.cnn_lstm_anomaly_model import AnomalyModel, AttentionLayer
from analysis.evaluation import ModelEvaluator

def test_workflow(logger):
    ### Class Reference
    merger = DataMerger()
    grouper = FileGrouping()
    loader = TracingBodyDataLoader()
    
    ### Create Dataset
    merged_df = merger.merged_df()
    merged_df = grouper.filegroup(merged_df)
    merged_df = merged_df[merged_df['group'] == 0]
    
    cleaned_df = loader.trace_unpacker(merged_df)
    cleaned_df["timestamp_seconds"] = pd.to_datetime(cleaned_df["sample_timestamp"], errors="coerce").astype(int) / 10**9
    cleaned_df["time_diff"] = cleaned_df["timestamp_seconds"].diff().fillna(0)
    cleaned_df["traces"] = cleaned_df["traces"].apply(
        lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x)
    )
    
    ### Model Training
    # --- Extract labels ---
    y = cleaned_df['anomaly_flag'].values
    logger.info(f"The label data is:\n {y}")
    
    # # --- Extract traces ---
    X_traces = np.stack(cleaned_df['traces'].apply(lambda x: np.array(x)).to_list())
    X_traces = (X_traces - X_traces.mean(axis=1, keepdims=True)) / \
    (X_traces.std(axis=1, keepdims=True) + 1e-8)
    n_traces = X_traces.shape[1]
    # --- Extract scalar features ---
    scalar_cols = cleaned_df.select_dtypes(include=[np.number]).columns.drop(['anomaly_flag','group'])
    X_scalars = cleaned_df[scalar_cols].values
    X_scalars = (X_scalars - X_scalars.mean(axis=0)) / \
    (X_scalars.std(axis=0) + 1e-8)
    X_combined = np.concatenate([X_traces,X_scalars], axis =1)
    
    scale = DataPreparation(test_size=0.2, random_state=42)
    X_resampled, y_resampled = scale.apply_smote(X_combined, y, sampling_ratio=0.6, k_neighbors=2)
    logger.info(f"X_combined resampled shape:{X_resampled.shape}")
    logger.info(f"y_resampled shape:{y_resampled.shape}")
    # --- Separate combined features back into traces and scalar features ---
    X_traces_resampled = X_resampled[:, :n_traces]
    X_scalars_resampled = X_resampled[:, n_traces:]
    
    # # Check the new class distribution:
    logger.info("Resampled class distribution:")
    logger.info(pd.Series(y_resampled).value_counts())
    
    #X_train, X_test, y_train, y_test = scale.split_data(X_resampled, y_resampled)
    X_trace_train, X_trace_test, X_scalar_train, X_scalar_test, y_train, y_test = train_test_split(
    X_traces_resampled, X_scalars_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
    
    # --- Load pre-trained model ---
    dummy_model = AnomalyModel(None, None)  # just to access the loss function
    custom_objects = {
        'AttentionLayer': AttentionLayer,
        'loss': dummy_model.focal_loss(gamma=2.0, alpha=0.25)  # or whatever values you used
    }
    try:
        model = load_model(
            '/w/data_science-sciwork24/ODU_CAPSTONE_2025/cnn_lstm/best_model.keras',
            custom_objects=custom_objects,
            compile=False
        )
        logger.info("Model loaded successfully for testing.")
    except Exception as e:
        logger.error(f"Error loading the model:{e}")
        return
    
    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

    # --- Evaluate the model ---
    evaluator = ModelEvaluator()
    y_pred_prob, y_pred_class, cm = evaluator.evaluate_model(
        model, X_trace_test, X_scalar_test, y_test, threshold=0.5)
    evaluator.export_metrics_pdf(cm, y_test, y_pred_prob, y_pred_class,
                                 pdf_filename='test_metrics_report.pdf')
    logger.info("Testing workflow finished.")