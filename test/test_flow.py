# test_flow.py
import logging
import numpy as np
from data_preparation.data_prep import DataPreparation
from analysis.evaluation import ModelEvaluator
from tensorflow.keras.models import load_model

def test_workflow(logger):
    logger.info("Starting testing workflow...")

    X = np.random.rand(50, 20)
    y = np.random.randint(0, 2, size=50)

    prep = DataPreparation(test_size=0.2, random_state=42)
    X_scaled = prep.scale_features(X)
    X_resampled, y_resampled = prep.apply_smote(X_scaled, y, sampling_ratio=0.2, k_neighbors=2)
    X_test = X_resampled
    y_test = y_resampled

    n_features = X_test.shape[1]
    n_trace = n_features // 2
    n_scalar = n_features - n_trace
    X_test_trace = X_test[:, :n_trace].reshape(-1, n_trace, 1)
    X_test_scalar = X_test[:, n_trace:]

    try:
        model = load_model('final_anomaly_detector.keras', compile=False)
        logger.info("Model loaded successfully for testing.")
    except Exception as e:
        logger.error("Error loading the model: %s", e)
        return

    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
    evaluator = ModelEvaluator()
    y_pred_prob, y_pred_class, cm = evaluator.evaluate_model(
        model, X_test_trace, X_test_scalar, y_test, threshold=0.5)
    evaluator.export_metrics_pdf(cm, y_test, y_pred_prob, y_pred_class,
                                 pdf_filename='test_metrics_report.pdf')
    logger.info("Testing workflow finished.")
