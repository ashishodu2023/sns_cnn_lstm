# train_flow.py
import logging
import datetime
import numpy as np
from data_parser.bpm_parser import BPMDataConfig  
from data_parser.dcm_config import DCMDatConfig     
from data_preparation.data_prep import DataPreparation
from model.anomaly_model import AnomalyModel
from analysis.evaluation import ModelEvaluator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from data_preparation.data_preprocessor import DataPreprocessor

def train_workflow(logger):
    logger.info("Starting training workflow...")

    # For example, you can now use BPMDataConfig to load beam parameters:
    bpm_config = BPMDataConfig()
    # Depending on your implementation, you might call:
    beam_config_df = bpm_config.update_beam_config(
        # Pass your DataFrame here...
    )
    # Similarly, you can instantiate and use DCMDatConfig as needed:
    dcm_config = DCMDatConfig()
    filtered_normal_files, filtered_anomaly_files = dcm_config.get_sep_filtered_files()
    
# Assuming merged_df is your DataFrame to be cleaned
    preprocessor = DataPreprocessor(merged_df)
    print("NaN values before removal:\n", preprocessor.check_nan())
    preprocessor.remove_nan()
    print("Duplicate rows before removal:", preprocessor.check_duplicates())
    preprocessor.remove_duplicates()
    print("Outliers detected:\n", preprocessor.check_outliers())
    preprocessor.remove_outliers()
    cleaned_df = preprocessor.convert_float64_to_float32().get_dataframe()


    # Continue with your data preparation (dummy data used here)
    X = np.random.rand(100, 20)
    y = np.random.randint(0, 2, size=100)

    prep = DataPreparation(test_size=0.2, random_state=42)
    X_scaled = prep.scale_features(X)
    X_resampled, y_resampled = prep.apply_smote(X_scaled, y, sampling_ratio=0.2, k_neighbors=2)
    X_train, X_test, y_train, y_test = prep.split_data(X_resampled, y_resampled)

    n_features = X_train.shape[1]
    n_trace = n_features // 2
    n_scalar = n_features - n_trace

    X_train_trace = X_train[:, :n_trace].reshape(-1, n_trace, 1)
    X_train_scalar = X_train[:, n_trace:]
    X_test_trace = X_test[:, :n_trace].reshape(-1, n_trace, 1)
    X_test_scalar = X_test[:, n_trace:]

    model_builder = AnomalyModel(
        input_trace_shape=(n_trace, 1),
        input_scalar_shape=(n_scalar,),
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
    model.fit([X_train_trace, X_train_scalar], y_train,
              epochs=10, batch_size=16, validation_split=0.2,
              callbacks=[tensorboard_callback, early_stop, checkpoint])
    logger.info("Model training completed.")

    evaluator = ModelEvaluator()
    model.load_weights('best_model.keras')
    y_pred_prob, y_pred_class, cm = evaluator.evaluate_model(
        model, X_test_trace, X_test_scalar, y_test, threshold=0.5)
    evaluator.export_metrics_pdf(cm, y_test, y_pred_prob, y_pred_class,
                                 pdf_filename='train_metrics_report.pdf')
    logger.info("Training workflow finished.")
