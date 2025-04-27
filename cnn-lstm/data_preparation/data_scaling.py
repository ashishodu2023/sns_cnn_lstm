#!/usr/bin/env python
# coding: utf-8

# data_preparation/data_prep.py
import logging
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class DataPreparation:
    """
    Handles data merging, cleaning, feature scaling, SMOTE oversampling,
    and train/test splitting.
    """
    def __init__(self, test_size=0.2, random_state=42):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.test_size = test_size
        self.random_state = random_state
        self.length_of_waveform = 10000

    def apply_smote(self, X: np.ndarray, y: np.ndarray, sampling_ratio=0.2, k_neighbors=2):
        self.logger.info("Applying SMOTE oversampling...")
        smote = SMOTE(sampling_strategy=sampling_ratio, random_state=self.random_state, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled

    def split_data(self, X: np.ndarray, y: np.ndarray):
        self.logger.info("Splitting data into train and test sets...")
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )