#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import logging
# from utils.logger import Logger
from data_preparation.data_loader import DataLoad

class DataMerger:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        # loader = DataLoad()
    
    def merged_df(self):
        loader = DataLoad()
        prepared_settings = loader.prepared_settings()
        dcm =  loader.dcm()
        merged_df = pd.merge_asof(
            dcm.sort_values("timestamps"), 
            prepared_settings.sort_values("timestamps"), 
            on="timestamps", 
            direction="nearest"
        )
        merged_df = merged_df[(merged_df['ICS_Tim:Gate_BeamOn:RR'] >= 59.90) & (merged_df['ICS_Tim:Chop_Flavor1:BeamOn'] >= 850)]
        merged_df = merged_df.sort_values("timestamps")
        merged_df = merged_df.reset_index(drop=True)
        return merged_df


class FileGrouping:

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def shifted_df(self, df):
        subset_columns = df.columns.tolist()
        subset_columns.remove('anomaly_flag')
        subset_columns.remove('timestamps')
        subset_columns.remove('file')
        
        df_shifted = df[subset_columns].shift(1)

        mask = (df[subset_columns] == df_shifted).all(axis=1)
        mask_df = pd.DataFrame(mask, columns = ['mtch_flg'])
        df = df.join(mask_df)
        return df
        
    def parametergroup(self, df, columns, column_name='group'):
        df[column_name] = 0
        for i in range(1, len(df)):
            if df.iloc[i][columns] is np.True_:
                df.loc[i, column_name] = df.loc[i - 1, column_name]
            else:
                df.loc[i, column_name] = df.loc[i - 1, column_name] + 1
        return df

    def filegroup(self, df):
        dateframe = self.shifted_df(df)
        dataframe = self.parametergroup(dateframe, 'mtch_flg', 'group')
        dataframe = dataframe.drop(columns=['mtch_flg'])
        groups = pd.DataFrame(dataframe.groupby(['group'])['file'].count())
        groups = groups.rename(columns={'file': 'file_ct'})
        
        dataframe = pd.merge(dataframe, groups, on=['group'], how = 'left')
        dataframe = dataframe[dataframe['file_ct'] > 25]
        dataframe = dataframe.drop(columns=['file_ct'])
        return dataframe

