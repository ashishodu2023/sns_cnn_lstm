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

# Authors: Kishansingh Rajput

import numpy as np
import pandas as pd
from sns_rad.binary import ReadStorage
import os

def get_traces(filename, var_id='Trace2', begin=3600, shift=4000, data_type=0, alarm=0, bWidth_cut=900):
# entry -0: Upstream waveform cycle 1
# entry - 1: Downstream waveform
# enetry -2: Parameters for the above waveforms/cycle
# entry -3 Upstream waveform cycle 2
# entry - 4 : Downstream
# entry - 5 : Parameters
    
    file = ReadStorage(filename)
    if shift is None:
        end = None
    else:
        end = begin + shift
    traces = []
    timestamps = []
    
    # Miha's cut
    for record in file:
        if 'bWidth' in record['parameters'].keys():
            bWidth_val  = record['parameters']['bWidth']
            if bWidth_val < bWidth_cut:
                print('File: ', filename, '  bWidth below threshold:', bWidth_val)
                file.close()
                return np.array(traces), np.array(timestamps)

    if ".gz" in filename:
        for idx in range(2, len(file), 3):
            if alarm == 0 and int(file[idx]['parameters']['Alarm']) != 0:
                print("Non zero alarm value ("+str(file[idx]['parameters']['Alarm'])+") in a named normal file at: ", filename)
                try:
                    print("Next timestamp is "+str((file[idx+3]['timestamp']-file[idx]['timestamp']))+" sec apart")
                except:
                    print("This was the last sample in the file...")
                return [], []
            
            flag = False
            if alarm == -1:
                if int(file[idx]['parameters']['Alarm']) != 0:
                    flag = True
            else:
                if (int(file[idx]['parameters']['Alarm']) == alarm):
                    flag = True
            if flag:
                timestamp = file[idx]['timestamp']
                if timestamp != file[idx-1]['timestamp']:
                    print("Error in sample order in file: ", filepath)
                    continue
                    
                idx_start, idx_end = idx-2, idx
                if alarm != 0:
                    if idx == 2: # or int(file[idx-3]['parameters']['Alarm']) != 0:
                        print("Fault is first entry in file...continue: ", filename)
                        continue
                    else:
                        idx_start, idx_end = idx-5, idx-3
                    
                for i in range(idx_start, idx_end):
                    if var_id in file[i]['name']:
                        trace = np.copy(file[i]['value'][begin:end])
                        traces.append(trace)
                        timestamps.append(file[i]['timestamp'])
    else:
        for idx in range(len(file)):
            if var_id in file[idx]['name']:
                trace = np.copy(file[idx]['value'][begin:end])
                if 'Before' == file[idx]['tags'][0] and data_type == -1:
                    traces.append(trace)
                    timestamps.append(file[idx]['timestamp'])
                if 'Before' in file[idx]['tags'][0] and data_type == 0:
                    traces.append(trace)
                    timestamps.append(file[idx]['timestamp'])
                if 'During' in file[idx]['tags'][0] and data_type == 1:
                    traces.append(trace)
                    timestamps.append(file[idx]['timestamp'])
    file.close()
    return traces, timestamps