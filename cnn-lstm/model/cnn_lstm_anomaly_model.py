# model/anomaly_model.py
import logging
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Concatenate, Bidirectional, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


# ----------------------------
# Custom Attention Layer Definition
# ----------------------------
class AttentionLayer(Layer):
    """
    This layer computes a weighted sum of the input features across the time dimension,
    allowing the model to focus on the most informative time steps.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # input_shape: (batch_size, time_steps, features)
        self.W = self.add_weight(name="att_weight", 
                                 shape=(input_shape[-1], 1),
                                 initializer="normal",
                                 trainable=True)
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Compute attention scores
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        # Compute softmax weights across time steps
        a = tf.keras.backend.softmax(e, axis=1)
        # Multiply weights with the input features and sum over the time dimension
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# ----------------------------
# Updated AnomalyModel Class
# ----------------------------
class AnomalyModel:
    """
    Builds and compiles a CNN+Bidirectional LSTM+Attention model for anomaly detection.
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
        self.logger.info("Building the improved model...")

        # ----- CNN Branch for Trace Input ----- #
        # Use the input tensor provided by self.input_trace_shape.
        # First convolution layer with L2 regularization, followed by batch normalization, pooling, and dropout.
        x = Conv1D(filters=64,
                   kernel_size=5,
                   padding='same',
                   activation='relu',
                   kernel_regularizer=l2(0.001))(self.input_trace_shape)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)

        # Second convolution layer with increased filters.
        x = Conv1D(filters=128,
                   kernel_size=3,
                   padding='same',
                   activation='relu',
                   kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)

        # Bidirectional LSTM layer to capture temporal dependencies.
        # Set return_sequences=True for the attention mechanism.
        x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)
        # Apply attention to let the network focus on the most informative time steps.
        x = AttentionLayer()(x)
        x = Dropout(0.3)(x)

        # ----- Merge with Scalar Input ----- #
        # Combine the output of the trace branch with scalar input features.
        combined = Concatenate()([x, self.input_scalar_shape])
        z = Dense(64, activation='relu')(combined)
        z = Dropout(0.3)(z)
        output = Dense(1, activation='sigmoid')(z)

        # Build the model: inputs are the trace and scalar inputs.
        self.model = Model(inputs=[self.input_trace_shape, self.input_scalar_shape], outputs=output)
        self.logger.info("Improved model built successfully.")

    def build_model_old(self):
        self.logger.info("Building the model (old version)...")
        # Original model architecture (for reference)
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
        self.logger.info("Old model built successfully.")

    def compile_model(self, optimizer='SGD'):
        self.logger.info("Compiling the model...")
        self.model.compile(
            optimizer=optimizer,
            loss=self.focal_loss(gamma=self.gamma, alpha=self.alpha),
            metrics=['accuracy']
        )
        self.logger.info("Model compiled.")

