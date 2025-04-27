#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np 
import os

class DCMDataConfig:
    """
    Holds configuration for DCM dataset,
    e.g., dataset paths, anomaly codes, etc.
    """
    def __init__(self):
        self.dataset_loc = "/w/data_science-sciwork24/suf_sns/DCML_dataset_Sept2024"
        self.length_of_waveform = 10000
        self.filtered_normal_files = []
        self.filtered_anomaly_files = []
        self.traces = []
        self.timestamps = []
        # self.logger = Logger()

    def get_sep_filtered_files(self):
        """Identify normal and anomaly files from dataset_loc."""
        # self.logger.info("====== Inside the get_sep_filtered_files ======")
        subfolders = [f.path for f in os.scandir(self.dataset_loc) if f.is_dir()]
        for directory in subfolders:
            if "normal" in directory or "anomal" in directory:
                for root, _, files in os.walk(directory):
                    for file in files:
                        if ".gz" in file:
                            if 'normal' in directory:
                                self.filtered_normal_files.append(os.path.join(root, file))
                            elif "anomal" in directory:
                                self.filtered_anomaly_files.append(os.path.join(root, file))
        return self.filtered_normal_files, self.filtered_anomaly_files


class BPMDataConfig:
    """
    Holds configuration for beam parameter data,
    e.g., file paths, columns, rename mappings, etc.
    """

    def __init__(self):
        # self.logger = Logger()
        self.beam_param_parser_cfg = {"data_location": "/work/data_science/suf_sns/beam_configurations_data/hdf5_sept2024/"}
        self.beam_settings_prep_cfg = {
            "rescale": False,
            "beam_config": [
                'FE_IS:Match:TunerPos',
                'LEBT:Chop_N:V_Set',
 
                'LEBT:Chop_P:V_Set',
                'LEBT:Focus_1:V_Set',
                'LEBT:Focus_2:V_Set',
                'LEBT:Steer_A:V_Set',
                'LEBT:Steer_B:V_Set',
                'LEBT:Steer_C:V_Set',
                'LEBT:Steer_D:V_Set',
                'Src:Accel:V_Set',
                'Src:H2:Flw_Set',
                'Src:Ign:Pwr_Set',
                'Src:RF_Gnd:Pwr_Set',
                'ICS_Chop:RampDown:PW',
                'ICS_Chop:RampUp:PWChange',
                'ICS_MPS:Gate_Source:Offset',
                'ICS_Tim:Chop_Flavor1:BeamOn',
                'ICS_Tim:Chop_Flavor1:OnPulseWidth',
                'ICS_Tim:Chop_Flavor1:RampUp',
                'ICS_Tim:Chop_Flavor1:StartPulseWidth',
                'ICS_Tim:Gate_BeamRef:GateWidth',
                'ICS_Tim:Gate_BeamOn:RR'
            ]
        }
        self.beam_config = [
            'timestamps',
            'FE_IS:Match:TunerPos',
            'LEBT:Chop_N:V_Set',
            'LEBT:Chop_P:V_Set',
            'LEBT:Focus_1:V_Set',
            'LEBT:Focus_2:V_Set',
            'LEBT:Steer_A:V_Set',
            'LEBT:Steer_B:V_Set',
            'LEBT:Steer_C:V_Set',
            'LEBT:Steer_D:V_Set',
            'Src:Accel:V_Set',
            'Src:H2:Flw_Set',
            'Src:Ign:Pwr_Set',
            'Src:RF_Gnd:Pwr_Set',
            'ICS_Chop:RampDown:PW',
            'ICS_Chop:RampUp:PWChange',
            'ICS_MPS:Gate_Source:Offset',
            'ICS_Tim:Chop_Flavor1:BeamOn',
            'ICS_Tim:Chop_Flavor1:OnPulseWidth',
            'ICS_Tim:Chop_Flavor1:RampUp',
            'ICS_Tim:Chop_Flavor1:StartPulseWidth',
            'ICS_Tim:Gate_BeamRef:GateWidth',
            'ICS_Tim:Gate_BeamOn:RR'
        ]
        self.column_to_add = [
            'FE_IS:Match:TunerPos',
            'LEBT:Chop_N:V_Set',
            'LEBT:Chop_P:V_Set',
            'LEBT:Focus_1:V_Set',
            'LEBT:Focus_2:V_Set',
            'LEBT:Steer_A:V_Set',
            'LEBT:Steer_B:V_Set',
            'LEBT:Steer_C:V_Set',
            'LEBT:Steer_D:V_Set',
            'Src:Accel:V_Set',
            'Src:H2:Flw_Set',
            'Src:Ign:Pwr_Set',
            'Src:RF_Gnd:Pwr_Set',
            'ICS_Tim:Gate_BeamOn:RR',
            'ICS_Chop-RampDown-PW',
            'ICS_Chop-RampUp-PWChange',
            'ICS_Tim-Gate_BeamRef-GateWidth']

        self.rename_mappings = {
            'ICS_Chop-RampDown-PW': 'ICS_Chop:RampDown:PW',
            'ICS_Chop-RampUp-PWChange': 'ICS_Chop:RampUp:PWChange',
            'ICS_MPS-Gate_Source-Offset': 'ICS_MPS:Gate_Source:Offset',
            'ICS_Chop-BeamOn-Width': 'ICS_Tim:Chop_Flavor1:BeamOn',
            'ICS_Chop-BeamOn-PW': 'ICS_Tim:Chop_Flavor1:OnPulseWidth',
            'ICS_Chop-RampUp-Width': 'ICS_Tim:Chop_Flavor1:RampUp',
            'ICS_Chop-RampUp-PW': 'ICS_Tim:Chop_Flavor1:StartPulseWidth',
            'ICS_Tim-Gate_BeamRef-GateWidth': 'ICS_Tim:Gate_BeamRef:GateWidth'}

    def update_beam_config(self, beam_config_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns exist and rename if needed."""
        # self.logger.info("====== Inside the update_beam_config ======")
        # for col in self.column_to_add:
        #     if col not in beam_config_df.columns:
        #         beam_config_df[col] = np.nan
        # beam_config_df.rename(columns=self.rename_mappings, inplace=True)
        return beam_config_df