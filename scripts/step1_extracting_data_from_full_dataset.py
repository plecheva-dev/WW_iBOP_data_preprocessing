# This package (and overall the manuscript) only needs a subset of the full dataset, 
# containing the ASCs data from the raw samples, and their corresponding metadata.

import pandas as pd
import os
import shutil
from pathlib import Path

raw_data_full_dataset_path = "data/raw_data_full_dataset"
raw_data_subset_for_this_package_path = Path("data/raw_data_subset_for_this_package")
raw_data_subset_for_this_package_path.mkdir(parents=True, exist_ok=True)

# sample_metadata.csv
sample_metadata_df = pd.read_csv(os.path.join(raw_data_full_dataset_path, "sample_metadata.csv"))
sample_metadata_df = sample_metadata_df[sample_metadata_df["sample"].str.startswith("S")]
sample_metadata_df["sample"] = sample_metadata_df["sample"].str.replace("S", "s")
sample_metadata_df.to_csv(os.path.join(raw_data_subset_for_this_package_path, "sample_metadata.csv"))


# laboratory reference measurements
lab_measurements_df = pd.read_csv(os.path.join(raw_data_full_dataset_path, "3_laboratory_measurements", "sample_s01-s24.csv"))
lab_measurements_df["tss_mg_l"] = lab_measurements_df["ss_04u_mg_l"] + lab_measurements_df["ss_14u_mg_l"]
columns_to_keep = ["sample", "tur_raw_ntu", "tss_mg_l", "tp_raw_mg_l", "tdp_04u_mg_l", "toc_raw_mg_l", "doc_04u_mg_l", "tn_raw_mg_l", "tdn_04u_mg_l"]
lab_measurements_df = lab_measurements_df[columns_to_keep]
lab_measurements_df.to_csv(os.path.join(raw_data_subset_for_this_package_path, "lab_measurements.csv"))

# ACs metadata
metadata_acs_df = pd.read_csv(os.path.join(raw_data_full_dataset_path, "1_acs_runs", "metadata_acs.csv"))
metadata_acs_filtered = metadata_acs_df[metadata_acs_df["sample_name"].str.endswith("raw")]
metadata_acs_filtered = metadata_acs_filtered[metadata_acs_filtered["sample_name"].str.startswith("s")]

def detect_sample_type(sample_name):
    return "diw" if "diw" in sample_name else "sample"
metadata_acs_filtered["sample_type"] = metadata_acs_filtered["sample_name"].apply(detect_sample_type)

metadata_acs_filtered.to_csv(os.path.join(raw_data_subset_for_this_package_path, "metadata_acs.csv"))

# ACs runs
valid_runs = set(metadata_acs_filtered["run_nr"])
acs_runs_path = "data/raw_data_full_dataset/1_acs_runs/runs"
dest_dir = os.path.join(raw_data_subset_for_this_package_path, "1_acs_runs")

os.makedirs(dest_dir, exist_ok=True)

for file_name in os.listdir(acs_runs_path):
    try:
        run_nr = int(file_name[-3:])
    except ValueError:
        continue # Skip system files like .DS_Store or files that don't end in numbers
        
    if run_nr in valid_runs:
        source_file = os.path.join(acs_runs_path, file_name)
        dest_file = os.path.join(dest_dir, file_name)
        shutil.copy2(source_file, dest_file)