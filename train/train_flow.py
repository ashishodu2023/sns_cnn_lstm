# train_flow.py
import logging
import datetime
import numpy as np
<<<<<<< HEAD
from parser.bpm_parser import BPMDataConfig  
from parser.dcm_parser import DCMDatConfig     
=======
import pandas as pd
from parser.bpm_parser import BPMDataConfig  
from parser.dcm_parser  import DCMDatConfig     
>>>>>>> 806193830e63186b2e8707216c4a78c2a918c101
from data_preparation.data_scaling import DataPreparation
from data_preparation.data_preprocessor import DataPreprocessor
from model.cnn_lstm_anomaly_model import AnomalyModel
from analysis.evaluation import ModelEvaluator
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from data_preparation.data_preprocessor import DataPreprocessor
from utils.logger import Logger
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def train_workflow(logger):
    logger = Logger()
    logger.info("Starting training workflow...")

    # BPMDataConfig to load beam parameters:
    bpm_config = BPMDataConfig()
    bpm_df = pd.read_csv(bpm_config.beam_settings_data_path)
    bpm_df = bpm_df.drop("Unnamed: 0", axis=1, errors="ignore")
    bpm_df['timestamps'] = pd.to_datetime(bpm_df['timestamps'])
    bpm_df = bpm_config.update_beam_config(bpm_df)
    logger.info(f"Beam config DataFrame created with shape:{bpm_df.shape}")

    # DCMDatConfig as parameters:
    dcm_config = DCMDatConfig()
    filtered_normal_files, filtered_anomaly_files = dcm_config.GetSepFilteredFiles()
    data_prep = DataPreparation()
    dcm_normal = data_prep.process_files(file_list=filtered_normal_files, label='Sep24', flag_value=0, data_type=0)
    dcm_anormal = data_prep.process_files(file_list=filtered_anomaly_files, label="Sep24", flag_value=1, data_type=-1, alarm=48)
    dcm_df=pd.concat([dcm_normal, dcm_anormal], ignore_index=True)
<<<<<<< HEAD
    logger.info("DCM DataFrame created with shape: %s", dcm_df.shape)

    merged_df=DataPreparation.merge_data(bcm_df,dcm_df)
    logger.info("Merged DataFrame created with shape: %s", merged_df.shape)

    processed_df = merged_df[merged_df.columns[~merged_df.columns.isin(
    ['timestamps', 'traces'])]]
    logger.info(processed_df.head())
    logger.info("Anomaly count: %s",processed_df['anomoly_flag'].value_counts())
    
=======
    logger.info(f"Dcm config DataFrame created with shape:{dcm_df.shape}")
    #logger.info(f"Anomanly flag in DCM dataframe: {dcm_df['anomoly_flag'].values}")
    merged_df=data_prep.merge_data(dcm_df,bpm_df)
    logger.info(f"Merged dataset:\n {merged_df.head()}")
    logger.info(f"Anomanly flag in Merged dataframe: {merged_df['anomoly_flag'].values}")
    processed_df = merged_df[merged_df.columns[~merged_df.columns.isin(['timestamps', 'traces'])]]
    logger.info(f"Details of the processed dataset:{processed_df.head()}")
    logger.info(processed_df['anomoly_flag'].value_counts())
>>>>>>> 806193830e63186b2e8707216c4a78c2a918c101

    # Assuming merged_df is your DataFrame to be cleaned
    preprocessor = DataPreprocessor(processed_df)
    print("NaN values before removal:\n", preprocessor.check_nan())
    preprocessor.remove_nan()
    print("Duplicate rows before removal:", preprocessor.check_duplicates())
    preprocessor.remove_duplicates()
    print("Outliers detected:\n", preprocessor.check_outliers())
    preprocessor.remove_outliers()
    
    cleaned_df_nan = data_prep.clean_data(processed_df,merged_df)
    logger.info(cleaned_df_nan.head())
    logger.info(f"Cleaned dataframe created with shape: {cleaned_df_nan.shape}")
    cleaned_df = cleaned_df_nan.dropna(axis=1)
    logger.info(cleaned_df.head())
    # --- Extract labels ---
    y = cleaned_df['anomoly_flag'].values
    logger.info(f"The label data is:\n {y}")

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
    logger.info(f"X_combined resampled shape:{X_resampled.shape}")
    logger.info(f"y_resampled shape:{y_resampled.shape}")
    # --- Separate combined features back into traces and scalar features ---
    X_traces_resampled = X_resampled[:, :n_traces]
    X_scalars_resampled = X_resampled[:, n_traces:]

    # Check the new class distribution:
    logger.info("Resampled class distribution:")
    logger.info(pd.Series(y_resampled).value_counts())

    #X_train, X_test, y_train, y_test = scale.split_data(X_resampled, y_resampled)
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
    y_pred_prob, y_pred_class, cm = evaluator.evaluate_model(
        model, X_trace_test, X_scalar_test, y_test, threshold=0.5)
    evaluator.export_metrics_pdf(cm, y_test, y_pred_prob, y_pred_class,
                                 pdf_filename='train_metrics_report.pdf')
    logger.info("Training workflow finished.")
