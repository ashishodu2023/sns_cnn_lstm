# data_preparation/data_prep.py
import logging
import numpy as np
import pandas as pd
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

    def process_files(self, file_list, label, flag_value, data_type, alarm=None):
        """
        Loops through file_list, extracts traces/timestamps, and appends flags.
        This is an example using get_traces(...) placeholders.
        """
        self.logger.info("====== Inside process_files ======")
        traces, timestamps, flags, files = [], [], [], []

        for dcml in file_list[:50]:
            if alarm is not None:
                tmp_trace, tmp_timestamp = get_traces(
                    dcml, var_id="Trace2", begin=3000,
                    shift=self.length_of_waveform, data_type=data_type, alarm=alarm
                )
            else:
                tmp_trace, tmp_timestamp = get_traces(
                    dcml, var_id="Trace1", begin=3000,
                    shift=self.length_of_waveform, data_type=data_type
                )

            if not tmp_trace or not tmp_timestamp:
                print(f"Skipping {dcml} due to empty trace/timestamp")
                continue

            tmp_trace = np.array(tmp_trace)
            tmp_timestamp = np.array(tmp_timestamp)
            if len(tmp_trace) > 1:
                tmp_trace = tmp_trace[1:]
            if len(tmp_timestamp) > 1:
                tmp_timestamp = tmp_timestamp[1:]

            traces.extend(tmp_trace.tolist())
            timestamps.extend(tmp_timestamp.tolist())
            flags.extend([flag_value] * len(tmp_trace))
            files.extend([label] * len(tmp_trace))

        return pd.DataFrame({
            'anomoly_flag': flags,
            'file': files,
            'timestamps': timestamps,
            'traces': traces
        })


    def merge_data(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Merging dataframes using merge_asof...")
        merged_df = pd.merge_asof(
            df1.sort_values("timestamps"),
            df2.sort_values("timestamps"),
            on="timestamps",
            direction="nearest"
        )
        return merged_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Cleaning data (removing NaNs and duplicates)...")
        df = df.dropna().drop_duplicates()
        df = preprocessor.convert_float64_to_float32().get_dataframe()
        df["traces"] = merged_df["traces"]
        df["timestamps"] = merged_df["timestamps"]
        df["timestamp_seconds"] = pd.to_datetime(df["timestamps"], errors="coerce").astype(int) / 10**9
        df["time_diff"] = df["timestamp_seconds"].diff().fillna(0)
        df.drop(columns=['file'], inplace=True, axis=1)
        df["traces"] = df["traces"].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x))
        return df

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
