from custom_python_tools.acs_data_reader import get_acs_IOP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

dir_path = "../data/raw_data_subset_for_this_package/1_acs_runs"

file_list = [ff for ff in os.listdir(dir_path) if ff.startswith('run_21_ACS.')]
f= file_list[0]

metadata = pd.read_csv("../data/raw_data_subset_for_this_package/metadata_acs.csv")
           
df_arr_A, df_arr_C = get_acs_IOP(os.path.join(dir_path, f))
wav_A = np.array(df_arr_A.columns[1:].astype(float))
wav_C = np.array(df_arr_C.columns[1:].astype(float))

samplename = metadata.loc[metadata["run_nr"]==int(f[-3:]), "sample_name"]

fig, ax = plt.subplots(ncols=2)

# Unpack the index (idx) and the row Series (data)
for idx, data in df_arr_A.iterrows():
    # Slice the row's data from index 1 onward to match wav_A (84 elements)
    ax[0].plot(wav_A, data.iloc[1:])

for idx, data in df_arr_C.iterrows():
    # Slice the row's data from index 1 onward to match wav_C (84 elements)
    ax[1].plot(wav_C, data.iloc[1:])

plt.show()