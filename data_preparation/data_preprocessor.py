# data_preprocessor.py
import numpy as np
import pandas as pd

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def check_nan(self):
        nan_counts = self.df.isna().sum()
        return nan_counts[nan_counts > 0]

    def remove_nan(self):
        self.df.dropna(inplace=True)
        return self.df

    def check_duplicates(self):
        df_copy = self.df.applymap(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)
        return df_copy.duplicated().sum()

    def remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        return self.df

    def check_outliers(self):
        outlier_dict = {}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))].shape[0]
            if outliers > 0:
                outlier_dict[col] = outliers
        return outlier_dict

    def remove_outliers(self):
        for col in self.df.select_dtypes(include=[np.number]).columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self.df

    def convert_float64_to_float32(self):
        float64_cols = self.df.select_dtypes(include=['float64']).columns
        self.df[float64_cols] = self.df[float64_cols].astype('float32')
        return self.df

    def rename_columns(self, rename_dict):
        self.df.rename(columns=rename_dict, inplace=True)
        return self.df

    def get_dataframe(self):
        return self.df
