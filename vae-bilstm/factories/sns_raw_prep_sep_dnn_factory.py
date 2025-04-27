#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy.fftpack import fft
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
import argparse

# Jlab Packages
from data_utils import get_traces
from beam_settings_parser_hdf5 import BeamConfigParserHDF5
from beam_settings_prep import BeamConfigPreProcessor

# TensorFlow
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Bidirectional, RepeatVector, TimeDistributed, Lambda
)

# Class Packages
from utils.logger import Logger
from data_preparation.data_loader import TracingBodyDataLoader
from data_preparation.data_transformer import DataMerger, FileGrouping
from data_preparation.data_scaling import DataPreparation
from models.vae_bilstm import MyVAE
from visualization.plots import plot_and_save_anomalies

pd.options.display.max_columns = None
pd.options.display.max_rows = None

class SNSRawPrepSepDNNFactory:
    def __init__(self):
        self.logger = Logger()
        self.window_size = 100
        self.num_features = 51

    def create_vae_bilstm_model(self, latent_dim: int = 16) -> MyVAE:
        """Factory method to build the VAE-BiLSTM model."""
        self.logger.info("====== Inside create_vae_bilstm_model ======")
        model = MyVAE(
            window_size=self.window_size,
            num_features=self.num_features,
            latent_dim=latent_dim
        )
        return model

    def extract_trace_features(self, trace_row: np.ndarray) -> np.ndarray:
        """Downsample + basic stats + partial FFT, etc."""
        downsampled = trace_row  # from 10k to 500
        mean_val = np.mean(downsampled)
        std_val = np.std(downsampled)
        peak_val = np.max(downsampled)
        fft_val = np.abs(fft(downsampled)[1])
        # example: keep first 50 + stats
        return np.hstack([downsampled[:50], mean_val, std_val, peak_val, fft_val])

    def _prepare_final_df(self) -> (pd.DataFrame, list):
        self.logger.info("====== Inside _prepare_final_df ======")

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
        
        for col_to_drop in ["file", "anomaly_flag", "sample_timestamp", "group"]:
            if col_to_drop in cleaned_df.columns:
                cleaned_df.drop(columns=[col_to_drop], inplace=True, errors='ignore')

        # 4) Feature extraction + PCA
        trace_features = np.array(
            cleaned_df["traces"].apply(self.extract_trace_features).tolist()
        )
        pca = PCA(n_components=50)
        trace_features_pca = pca.fit_transform(trace_features)
        trace_feature_names = [f"PCA_Trace_{i}" for i in range(trace_features_pca.shape[1])]

        df_pca = pd.DataFrame(trace_features_pca, columns=trace_feature_names)
        df_final = pd.concat([cleaned_df.drop(columns=["traces"], errors="ignore"), df_pca], axis=1)

        return df_final, trace_feature_names

    # ---------------------------
    # TRAIN PIPELINE
    # ---------------------------
    def train_pipeline(
        self,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        latent_dim: int = 16,
        model_path: str = "vae_bilstm_model.weights.h5",
        tensorboard_logdir: str = "logs/fit"
    ):
        """
        1) Prepares final DataFrame (df_final)
        2) Builds & trains VAE-BiLSTM
        3) Saves model weights
        """
        self.logger.info("====== Inside train_pipeline ======")

        df_final, trace_feature_names = self._prepare_final_df()

        # Build model
        vae_model = self.create_vae_bilstm_model(latent_dim=latent_dim)
        _ = vae_model(tf.zeros((1, 100, 51)))
        vae_model.build((None, self.window_size, self.num_features))
        self.logger.info(vae_model.summary())
        # Compile
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        vae_model.compile(optimizer=optimizer, loss='mae')
        
        print(vae_model.summary())

        # Create rolling windows.
        # With window_size = 3, each window consists of 3 pulses.
        # For forecasting anomaly, we will later compute reconstruction error only on the last (nth) pulse.
        X_train_combined = []
        for i in range(self.window_size, len(df_final)):
            window = df_final.iloc[i - self.window_size : i][trace_feature_names + ["time_diff"]]
            X_train_combined.append(window.values)
        X_train_combined = np.array(X_train_combined, dtype=np.float32)
        X_train_combined = np.nan_to_num(X_train_combined)

        # Setup TensorBoard callback
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(tensorboard_logdir, time_str)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
        self.logger.info(f"TensorBoard logs will be saved to: {log_dir}")

        # Train the model (autoencoder reconstructs full window)
        history = vae_model.fit(
            X_train_combined, X_train_combined,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[tensorboard_callback]
        )

        # Save weights
        vae_model.save_weights(model_path)
        self.logger.info(f"Model saved to: {model_path}")
        self.logger.info("====== Training pipeline completed ======")

    # ---------------------------
    # PREDICT PIPELINE
    # ---------------------------
    def predict_pipeline(
        self,
        model_path: str = "vae_bilstm_model.weights.h5",
        threshold_percentile: float = 90.0
    ):
        """
        1) Prepares final DataFrame (df_final)
        2) Loads VAE-BiLSTM weights
        3) Computes reconstruction errors & anomalies
        """
        self.logger.info("====== Inside predict_pipeline ======")

        df_final, trace_feature_names = self._prepare_final_df()

        # Build the model architecture and perform a dummy forward pass to initialize weights.
        new_vae_model = MyVAE(window_size=self.window_size, num_features=self.num_features, latent_dim=16)
        new_vae_model.build((None, self.window_size, self.num_features))
        
        # dummy_input = tf.zeros((1, self.window_size, self.num_features), dtype=tf.float32)
        # _ = new_vae_model(dummy_input)
        
        new_vae_model.compile(optimizer='sgd', loss='mae') 
        weights_path = os.path.expanduser(model_path)
        new_vae_model.load_weights(weights_path)
        self.logger.info(f"Model weights loaded from: {model_path}")

        # Create test windows
        X_test_combined = []
        for i in range(self.window_size, len(df_final)):
            window = df_final.iloc[i - self.window_size : i][trace_feature_names + ["time_diff"]]
            X_test_combined.append(window.values)
        X_test_combined = np.array(X_test_combined, dtype=np.float32)
        X_test_combined = np.nan_to_num(X_test_combined)

        # Predict: new_vae_model predicts an entire window (3 pulses)
        X_pred = new_vae_model.predict(X_test_combined)
        # Compute reconstruction error only for the nth pulse (i.e. last timestep)
        reconstruction_errors = np.mean(np.abs(X_test_combined[:,-1,:] - X_pred[:,-1,:]), axis=1)
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        anomalies = reconstruction_errors > threshold

        df_anomalies = pd.DataFrame({
            "Timestamp": df_final["timestamps"].iloc[self.window_size:],
            "Reconstruction_Error": reconstruction_errors,
            "Anomaly": anomalies
        })

        self.logger.info(f"Top 20 anomalies (threshold={threshold:.4f} at {threshold_percentile} percentile):")
        self.logger.info(df_anomalies.sort_values(by="Reconstruction_Error", ascending=False).head(20))
        self.logger.info("====== Prediction pipeline completed ======")

        self.logger.info("====== Plotting and saving reconstruction error plots ======")
        plot_and_save_anomalies(
            df_anomalies, 
            threshold=threshold, 
            dist_filename="dist_plot.png", 
            time_filename="time_plot.png"
        )
        self.logger.info("====== Saved Plots (PNG) ======")