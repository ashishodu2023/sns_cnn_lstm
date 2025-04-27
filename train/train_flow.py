# train_flow.py
import logging
import datetime
import numpy as np
from parser.bpm_parser import BPMDataConfig  
from parser.dcm_parser import DCMDatConfig     
from data_preparation.data_scaling import DataPreparation
from model.anomaly_model import AnomalyModel
from analysis.evaluation import ModelEvaluator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from data_preparation.data_preprocessor import DataPreprocessor

def train_workflow(logger):
    logger.info("Starting training workflow...")

    # BPMDataConfig to load beam parameters:
    bpm_config = BPMDataConfig()
    bcm_df = pd.read_csv(bpm_config.beam_settings_data_path)
    bcm_df = bcm_df.drop("Unnamed: 0", axis=1, errors="ignore")
    bcm_df['timestamps'] = pd.to_datetime(bcm_df['timestamps'])
    bcm_df = bpm_config.update_beam_config(bcm_df)
    logger.info("Beam config DataFrame created with shape: %s", bcm_df.shape)

    # DCMDatConfig as parameters:
    dcm_config = DCMDatConfig()
    filtered_normal_files, filtered_anomaly_files = dcm_config.get_sep_filtered_files()
    dcm_normal = DataPreparation.process_files(filtered_normal_files, 'Sep24', 0, data_type=0)
    dcm_anormal = DataPreparation.process_files(filtered_anomaly_files, 'Sep24', 1, data_type=-1, alarm=48)
    dcm_df=pd.concat([dcm_normal, dcm_anormal], ignore_index=True)
    logger.info("DCM DataFrame created with shape: %s", dcm_df.shape)

    merged_df=DataPreparation.merge_data(bcm_df,dcm_df)
    logger.info("Merged DataFrame created with shape: %s", merged_df.shape)

    processed_df = merged_df[merged_df.columns[~merged_df.columns.isin(
    ['timestamps', 'traces'])]]
    logger.info(processed_df.head())
    logger.info("Anomaly count: %s",processed_df['anomoly_flag'].value_counts())
    

    # Assuming merged_df is your DataFrame to be cleaned
    preprocessor = DataPreprocessor(processed_df)
    print("NaN values before removal:\n", preprocessor.check_nan())
    preprocessor.remove_nan()
    print("Duplicate rows before removal:", preprocessor.check_duplicates())
    preprocessor.remove_duplicates()
    print("Outliers detected:\n", preprocessor.check_outliers())
    preprocessor.remove_outliers()
    
    cleaned_df = DataPreparation.clean_data(merged_df)
    logger.info("Cleaned dataframe created with shape: %s", cleaned_df.shape)

    # --- Extract labels ---
    y = cleaned_df['anomoly_flag'].values

    # --- Extract traces ---
    X_traces = np.stack(cleaned_df['traces'].apply(lambda x: np.array(x)).to_list())
    X_traces = (X_traces - X_traces.mean(axis=1, keepdims=True)) / \
    (X_traces.std(axis=1, keepdims=True) + 1e-8)
    n_traces = X_traces.shape[1]
    # --- Extract scalar features ---
    scalar_cols = cleaned_df.select_dtypes(include=[np.number]).columns.drop(['anomoly_flag'])
    X_scalars = cleaned_df[scalar_cols].values
    X_scalars = (X_scalars - X_scalars.mean(axis=0)) / \
    (X_scalars.std(axis=0) + 1e-8)
    X_combined = np.concatenate([X_traces,X_scalars], axis =1)
    
    scale = DataPreparation(test_size=0.2, random_state=42)
    X_resampled, y_resampled = scale.apply_smote(X_combined, y, sampling_ratio=0.2, k_neighbors=2)
    logger.info("X_combined resampled shape:", X_resampled.shape)
    logger.info("y_resampled shape:", y_resampled.shape)
    # --- Separate combined features back into traces and scalar features ---
    X_traces_resampled = X_resampled[:, :n_traces]
    X_scalars_resampled = X_resampled[:, n_traces:]

    # Check the new class distribution:
    logger.info("Resampled class distribution:")
    logger.info(pd.Series(y_resampled).value_counts())

    X_train, X_test, y_train, y_test = scale.split_data(X_resampled, y_resampled)

     
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
