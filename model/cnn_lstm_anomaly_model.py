# model/anomaly_model.py
import logging
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Concatenate
)
from tensorflow.keras.models import Model

class AnomalyModel:
    """
    Builds and compiles a CNN+LSTM model for anomaly detection.
    """
    def __init__(self, input_trace_shape, input_scalar_shape, gamma=0.25, alpha=0.25):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.input_trace_shape = input_trace_shape
        self.input_scalar_shape = input_scalar_shape
        self.gamma = gamma
        self.alpha = alpha
        self.model = None

    def focal_loss(self, gamma=2.0, alpha=0.25):
        def loss(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            return -tf.keras.backend.mean(alpha * tf.keras.backend.pow((1. - pt), gamma) *
                                          tf.keras.backend.log(pt))
        return loss

    def build_model(self):
        self.logger.info("Building the model...")
        #trace_input = Input(shape=self.input_trace_shape, name="trace_input")
        #scalar_input = Input(shape=self.input_scalar_shape, name="scalar_input")

        # CNN + LSTM for trace input
        x = Conv1D(64, kernel_size=5, padding='same', activation='relu')(self.input_trace_shape)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.3)(x)
        x = LSTM(64)(x)
        x = Dropout(0.3)(x)

        combined = Concatenate()([x, self.input_scalar_shape])
        z = Dense(64, activation='relu')(combined)
        z = Dropout(0.3)(z)
        output = Dense(1, activation='sigmoid')(z)

        self.model = Model(inputs=[self.input_trace_shape, self.input_scalar_shape], outputs=output)
        self.logger.info("Model built successfully.")

    def compile_model(self, optimizer='SGD'):
        self.logger.info("Compiling the model...")
        self.model.compile(
            optimizer=optimizer,
            loss=self.focal_loss(gamma=self.gamma, alpha=self.alpha),
            metrics=['accuracy']
        )
        self.logger.info("Model compiled.")
