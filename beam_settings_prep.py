# Copyright (c) 2022, Jefferson Science Associates, LLC. All Rights Reserved. Redistribution
# and use in source and binary forms, with or without modification, are permitted as a
# licensed user provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this
#    list of conditions and the following disclaimer in the documentation and/or other
#    materials provided with the distribution.
# 3. The name of the author may not be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# This material resulted from work developed under a United States Government Contract.
# The Government retains a paid-up, nonexclusive, irrevocable worldwide license in such
# copyrighted data to reproduce, distribute copies to the public, prepare derivative works,
# perform publicly and display publicly and to permit others to do so.
#
# THIS SOFTWARE IS PROVIDED BY JEFFERSON SCIENCE ASSOCIATES LLC "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
# JEFFERSON SCIENCE ASSOCIATES, LLC OR THE U.S. GOVERNMENT BE LIABLE TO LICENSEE OR ANY
# THIRD PARTIES FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# Standard Library Imports

# Script: Beam configuration pre-processing
# Authors: Kishansingh Rajput

# Third party imports
import pandas as pd
import os
import pickle
import numpy as np
import datetime



class BeamConfigPreProcessor:
    """Beam Configuration pre-processing.

    Parameters
    ----------
    JDSTDataPrep : Inherit from core dataprep class
        
    """
    def __init__(self, config, name=None):
        """Initialize the object with configuration file

        Parameters
        ----------
        config : dict
            A python dictionary containing pre-processor settings

        """
        self.name = name
        self.rescale = config['rescale'] if 'rescale' in config else False
        self.scaling_min_max = config['scaling_min_max'] if 'scaling_min_max' in config else None
        self.output_location = config['output_location'] if 'output_location' in config else None
        self.beam_config_names = config['beam_config'] if 'beam_config' in config else None
        if self.output_location is not None:
            os.makedirs(self.output_location, exist_ok=True)
            
    
    def convert_to_df(self, key, value):
        """Convert a dictionary to dataframe

        Parameters
        ----------
        key : str
            Beam parameter name
        value : dict
            A dictionary with two coloumns namely timestamps and values respectively

        Returns
        -------
        pandas DataFrame
            returns a data frame built from the input
        """
        df = pd.DataFrame(value)
        df.columns = ['timestamps', key]
        df[key] = pd.to_numeric(df[key], errors='coerce')
        return df
    
    def merge_df(self, master_df, df):
        master_df = master_df.merge(df.sort_values(by='timestamps'), how='outer', on='timestamps', sort=True)
        master_df.fillna(method='ffill', inplace=True)
        
        return master_df
    
    def rescale_df(self, df):
        if not self.rescale:
            return df
        
        cols = [col for col in df.columns if col not in ['timestamps', 'index', 'Index', 'Timestamps']]
        if self.scaling_min_max is None:
            rescale_min = df[cols].min()
            rescale_max = df[cols].max()
        else:
            rescale_min, rescale_max = [], []
            for col in cols:
                rescale_min.append(self.scaling_min_max[col][0])
                rescale_max.append(self.scaling_min_max[col][1])
            rescale_min, rescale_max = np.array(rescale_min), np.array(rescale_max)
            
        df[cols] = (df[cols] - rescale_min) / (rescale_max - rescale_min)
        
        return df
    
    def run(self, data, save=False, plot=False):
        master_df = None
        keys = list(data.keys())
        if self.beam_config_names is not None:
            print(self.beam_config_names)
            keys = self.beam_config_names
        for key in keys:
            df = self.convert_to_df(key, data[key])
            if master_df is None:
                master_df = df.sort_values(by='timestamps')
            else:
                master_df = self.merge_df(master_df, df)
                
        master_df.dropna(inplace=True)
        # master_df['timestamps'] = pd.to_datetime(master_df['timestamps'], unit='ns')
        # Manual shifting of one hour to account for daylight saving shift - will be fixed in the beam configuration data in next version and then manual shifting will be removed
        master_df['timestamps'] = master_df['timestamps'].apply(lambda x: datetime.datetime.fromtimestamp(x/10**9, tz=None)-datetime.timedelta(hours=1))
        master_df.reset_index(inplace=True, drop=True)
        if plot:
            self.plot_config(master_df)
        
        self.master_df = self.rescale_df(master_df)
        print("Length of beam param df: ", len(self.master_df))

        if save:  
            self.save_data()
        configurations = self._make_configurations()

        return self.master_df, configurations
    
    def _make_configurations(self):
        configurations = {
            self.name:{
                "rescale": self.rescale,
                "scaling_min_max": self.scaling_min_max,
                "beam_config": self.beam_config_names
            }
        }
        return configurations
    
    def save_data(self):
        if self.output_location is not None:
            configurations = self._make_configurations()
            with open(os.path.join(self.output_location, "beam_par_preprocessor_config.txt"), "wb") as f:
                pickle.dump(configurations, f)

            with open(os.path.join(self.output_location, "pre_processed_beam_config.pkl"), "wb") as f:
                pickle.dump(self.master_df, f)

    def reverse(self):
        print("reverse function is not implemented for the beam config pre-processor, returns None")
        return None
        