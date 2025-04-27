from data_utils import get_traces
from beam_settings_parser_hdf5 import BeamConfigParserHDF5
from beam_settings_prep import BeamConfigPreProcessor
import pandas as pd
import numpy as np 
import os
from utils.logger import Logger

class DCMDatConfig:

    def __init__(self):
        self.dataset1_loc = "/work/data_science/suf_sns/DCM_Errant/"
        self.dataset2_loc = "/w/data_science-sciwork24/suf_sns/DCML_dataset_Sept2024"
        self.start_date = 20220218
        self.end_date = 20220318
        self.anomaly_type = "00110000"  # --  48
        self.length_of_waveform = 10000
        self.exclude_dates = [20220220, 20220221, 20220222, 20220223,
                              20220301, 20220308, 20220309, 20220315]  # Fixed date

        self.filtered_normal_files = []
        self.filtered_anomaly_files = []
        self.filtered_normal_files2 = []
        self.filtered_anomaly_files2 = []
        self.traces = []
        self.timestamps = []

    def GetSepFilteredFiles(self):
        subfolders = [f.path for f in os.scandir(
            self.dataset2_loc) if f.is_dir()]
        for directory in subfolders:
            if "normal" in directory or "anomal" in directory:
                for root, _, files in os.walk(directory):
                    for file in files:
                        if ".gz" in file:
                            if 'normal' in directory:
                                self.filtered_normal_files2.append(
                                    os.path.join(root, file))
                            elif "anomal" in directory:
                                self.filtered_anomaly_files2.append(
                                    os.path.join(root, file))

        logger.info('Number of available normal files:%s',
              len(self.filtered_normal_files2))
        logger.info('Number of available anomaly files: %s',
              len(self.filtered_anomaly_files2))
        return self.filtered_normal_files2, self.filtered_anomaly_files2


    def GetTracesAndTs(self, filtered_files):
        index = np.random.randint(0, len(filtered_files))
        filepath = filtered_files[index]
        print(index)
        try:
            self.traces, self.timestamps = get_traces(
                filepath, var_id="Trace2", begin=3000, shift=self.length_of_waveform, data_type=0)
        except Exception as e:
            print("Error in reading the file: ", filepath)
            print("Error:", e)
        return self.traces, self.timestamps
