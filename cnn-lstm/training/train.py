#!/usr/bin/env python
# coding: utf-8

### Python Packages
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

### Class Packages
from utils.logger import Logger
from data_preparation.data_loader import TracingBodyDataLoader
from data_preparation.data_transformer import DataMerger, FileGrouping
from data_preparation.data_scaling import DataPreparation
from model.cnn_lstm_anomaly_model import AnomalyModel
from analysis.evaluation import ModelEvaluator

def train_workflow(logger):
    ### Class Reference
    merger = DataMerger()
    grouper = FileGrouping()
    loader = TracingBodyDataLoader()
    
    ### Create Dataset
    # --- Combine BPM & DCM into single dataframe ---
    merged_df = merger.merged_df()
    merged_df = grouper.filegroup(merged_df)
    merged_df = merged_df[merged_df['group'] == 0]
    # --- Clean dataframe ---
    cleaned_df = loader.trace_unpacker(merged_df)
    cleaned_df["timestamp_seconds"] = pd.to_datetime(cleaned_df["sample_timestamp"], errors="coerce").astype(int) / 10**9
    cleaned_df["time_diff"] = cleaned_df["timestamp_seconds"].diff().fillna(0)
    cleaned_df["traces"] = cleaned_df["traces"].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x))
    
    ### Model Training
    # --- Extract labels ---
    y = cleaned_df['anomaly_flag'].values
    logger.info(f"The label data is:\n {y}")
    
    # --- Extract traces ---
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
    
    # --- Apply SMOTE ---
    scale = DataPreparation(test_size=0.2, random_state=42)
    X_resampled, y_resampled = scale.apply_smote(X_combined, y, sampling_ratio=0.6, k_neighbors=2)
    logger.info(f"X_combined resampled shape:{X_resampled.shape}")
    logger.info(f"y_resampled shape:{y_resampled.shape}")
    
    # --- Separate combined features back into traces and scalar features ---
    X_traces_resampled = X_resampled[:, :n_traces]
    X_scalars_resampled = X_resampled[:, n_traces:]
    
    # ---Check the new class distribution ---
    logger.info("Resampled class distribution:")
    logger.info(pd.Series(y_resampled).value_counts())
    
    # --- Separate combined features back into traces and scalar features ---
    X_trace_train, X_trace_test, X_scalar_train, X_scalar_test, y_train, y_test = train_test_split(
    X_traces_resampled, X_scalars_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
    
    # --- Model Inputs ---
    trace_input = Input(shape=(X_traces.shape[1], 1), name="trace_input")
    scalar_input = Input(shape=(X_scalars.shape[1],), name="scalar_input") 
    model_builder = AnomalyModel(
        input_trace_shape=trace_input,
        input_scalar_shape=scalar_input,
        gamma=0.25,
        alpha=0.25
    )
    model_builder.build_model()
    model_builder.compile_model(optimizer='SGD')
    model = model_builder.model
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)
    
    logger.info("Starting model training...")
    model.fit([X_trace_train[..., None], X_scalar_train], y_train, epochs=5, batch_size=32, validation_split=0.2,
                    callbacks=[early_stop, checkpoint, tensorboard_callback])
    logger.info("Model training completed.")
    
    evaluator = ModelEvaluator()
    model.load_weights('best_model.keras')
    # y_pred_prob, y_pred_class, cm = evaluator.evaluate_model(
    #     model, X_trace_test, X_scalar_test, y_test, threshold=0.5)
    # evaluator.export_metrics_pdf(cm, y_test, y_pred_prob, y_pred_class,
    #                              pdf_filename='train_metrics_report.pdf')
    y_pred_prob, y_pred_class, cm = evaluator.evaluate_model(
        model, X_trace_train, X_scalar_train, y_train, threshold=0.5)
    evaluator.export_metrics_pdf(cm, y_train, y_pred_prob, y_pred_class,
                                 pdf_filename='train_metrics_report.pdf')
    logger.info("Training workflow finished.")