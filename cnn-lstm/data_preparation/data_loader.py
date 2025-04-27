#!/usr/bin/env python
# coding: utf-8

# Standard Packages
import pandas as pd
import datetime
from datetime import datetime, timedelta
import sys
import logging
# JLab Packages
from beam_settings_parser_hdf5 import BeamConfigParserHDF5
from beam_settings_prep import BeamConfigPreProcessor
from data_utils import get_traces
# Model Packages
from utils.logger import Logger
from parser.configs import DCMDataConfig, BPMDataConfig

class BeamDataLoader:
    """
    Loads and merges BPM beam configuration data.
    """

    def __init__(self, config: BPMDataConfig):
        self.config = config
        # self.logger = Logger()

    def load_beam_config_df(self) -> pd.DataFrame:
        """Loads beam config CSV and updates columns."""
        config = BPMDataConfig()
        # self.logger.info("====== Inside load_beam_config_df ======")
        parser = BeamConfigParserHDF5(config.beam_param_parser_cfg)
        data, _ = parser.run()
        prep = BeamConfigPreProcessor(config.beam_settings_prep_cfg)
        beam_config_df, run_cfg = prep.run(data) 
        return beam_config_df

class TracingHeaderDataLoader:

    def __init__(self):
        # self.logger = Logger()
        self.file_date = []
        self.flag = []

    def create_dcm_data(self) -> pd.DataFrame:
        dcm_config = DCMDataConfig()
        normal_files, anomaly_files = dcm_config.get_sep_filtered_files()
        return normal_files, anomaly_files

class DataLoad:

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def prepared_settings(self):
        beam_loader = BeamDataLoader(BPMDataConfig())
        prepared_settings = beam_loader.load_beam_config_df()
        return prepared_settings
        
    def dcm(self):
        tracing_loader = TracingHeaderDataLoader()
        normal_files, anomaly_files = tracing_loader.create_dcm_data()
        file_date = []
        flag = []
        for file in normal_files:
            file_time = file[-32:-12]
            file_time = file_time.replace("_","")
            file_time = datetime.strptime(file_time,'%Y%m%d%H%M%S.%f')
            file_date.append(file_time)
            flag.append(0)
        
        for file in anomaly_files:
            file_time = file[-32:-12]
            file_time = file_time.replace("_","")
            file_time = datetime.strptime(file_time,'%Y%m%d%H%M%S.%f')
            file_date.append(file_time)
            flag.append(1)

        dcm = pd.DataFrame({'anomaly_flag':flag, 'timestamps':file_date, 'file':normal_files+anomaly_files})
        return dcm

class TracingBodyDataLoader:

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def trace_unpacker(self, df):
        traces = []
        timestamps = []
        file = []
        length_of_waveform = 10000
        for filepath in df['file']:
            if filepath[-40:-39] == 's':
                try:
                    trace, timestamp = get_traces(filepath, var_id="Trace2", begin=3000, shift=length_of_waveform, data_type=-1, alarm=48)
                    for sample in trace:
                        trace = sample
                        traces.append(trace)
                        file.append(filepath)
                    for stamp in timestamp:
                        timestamps.append(stamp)
                except:
                    pass
            else:
                try:
                    trace, timestamp = get_traces(filepath, var_id="Trace2", begin=3000, shift=length_of_waveform, data_type=-1, alarm=0) 
                    for sample in trace:
                        trace = sample
                        traces.append(trace)
                        file.append(filepath)
                    for stamp in timestamp:
                        timestamps.append(stamp)
                except:
                    pass
        trace_df = pd.DataFrame({'file':file, 'traces':traces, 'sample_timestamp': timestamps})
        sns = pd.merge(df, trace_df, on=['file'], how = 'inner')
        return sns