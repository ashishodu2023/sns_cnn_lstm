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

# Script: Parser for beam configuration parameters from Archiver (hdf5 format)
# Authors: Kishansingh Rajput

# Standard imports
import os
import sys
import numpy as np
import h5py
import pickle


class BeamConfigParserHDF5:
    """Parser for beam configuration data from archiver

    """
    def __init__(self, config, name=None):
        """Initialize the beam config parser

        Args:
            config: A dictionary containing parser settings
        """
        self.name = name
        self.data_location = config['data_location'] if 'data_location' in config else None
        if self.data_location is None:
            print("'data_location' entry is not provided in config, please provide the location of raw data ",
                  "to parse from for the key 'data_location' in config...")
            sys.exit()

        self.output_directory = config['output_location'] if 'output_location' in config else None
        
        if self.output_directory is not None:
            os.makedirs(self.output_directory, exist_ok=True)
        

    def read_hdf5_file(self, filepath):
        file = h5py.File(filepath)
        d = dict()
        for key,val in file.items():
            if type(val) == h5py._hl.dataset.Dataset:
                d[key] = np.array(val)
            else:
                d[key] = read_hdf5_file(val)
        return d
    
    def parse_file(self, filepath):
        data = self.read_hdf5_file(filepath)
        return data
    
    def parse_dir(self):
        files = os.listdir(self.data_location)
        master_dict = None
        for file in files:
            if file[-5:] != '.hdf5':
                print("Provided file is not hdf5 format, skipping: ", file)
                continue

            filepath = os.path.join(self.data_location, file)
            data = self.parse_file(filepath)
            if master_dict is None:
                master_dict = data
            else:
                for key in data:
                    if key in master_dict:
                        # Skip if the same config is already read
                        continue
                    master_dict[key] = data[key]
        return master_dict
                    
    
    def run(self, save=False):
        if os.path.isdir(self.data_location):
            self.parsed_data = self.parse_dir()
        else:
            self.parsed_data = self.parse_file(self.data_location) 

        configurations = self._make_configurations()
        print("BeamParamParser: Number of samples parsed")
        for key in self.parsed_data:
            print(key, len(self.parsed_data[key]))

        if save:
            self.save_data()
        
        return self.parsed_data, configurations
    
    def _make_configurations(self):
        """
        """
        configurations = {
            self.name: {
                'data_location': self.data_location,
                'output_location': self.output_directory
            }
        }

        return configurations
        
    
    def save_data(self):

        if self.output_directory is not None:
            configurations = self._make_configurations()
            with open(os.path.join(self.output_directory, "beam_param_parser_config.txt"), "wb") as f:
                pickle.dump(configurations, f)

            with open(os.path.join(self.output_directory, "parsed_beam_param.pkl"), "wb") as f:
                pickle.dump(self.parsed_data, f)










