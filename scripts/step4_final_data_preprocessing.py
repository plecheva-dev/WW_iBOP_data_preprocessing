
from custom_python_tools.acs_utils import get_runs_list_from_samplename_and_metadata
from custom_python_tools.acs_utils import handle_exception_when_run_not_working
from custom_python_tools.acs_utils import concatenate_df
from custom_python_tools.acs_data_reader import get_acs_IOP
from custom_python_tools.acs_outlier_detection_pipeline import apply_outlier_detection_pipeline
from custom_python_tools.acs_scattering_coefficient_calculation import calculate_scattering_spectrum
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

# utils for this script only
def process_and_save_spectra(spectra_dict, columns, output_path, output_filename):
    """
    Converts a raw spectra dictionary to a DataFrame, parses sample metadata,
    and saves the cleaned DataFrame to a CSV (name=output_filename) in the output_path folder.
    """
    # 1. Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(spectra_dict, orient='index', columns=columns)
    
    # 2. Reset index and rename to 'full_sample_name'
    df.reset_index(inplace=True)
    df.rename(columns={"index": "full_sample_name"}, inplace=True)
    
    # 3. Apply the parsing function you created earlier
    df = preprocess_final_spectra_df(df)
    
    # 4. Save to CSV
    df.to_csv(os.path.join(output_path, output_filename), index_label="index")
    return df

def preprocess_final_spectra_df(spectra_df):
    # 1. Split the 'full_sample_name' column
    # This creates a temporary DataFrame with two columns
    metadata = spectra_df['full_sample_name'].str.split('_', expand=True)

    # 2. Assign the new columns to your original DataFrame
    spectra_df['sample'] = metadata[0]
    spectra_df['preprocessing'] = metadata[1]

    # 3. Reorder columns to put the new ones at the front
    # We grab the new names, then every name that isn't those two
    cols = ['sample', 'preprocessing'] + [c for c in spectra_df.columns if c not in ['sample', 'preprocessing']]
    spectra_df = spectra_df[cols]
    return spectra_df

### main script ###

def main():
    acs_dir_path = "data/raw_data_subset_for_this_package/1_acs_runs"
    data_output_path = "data/processed"
    figures_output_path = Path("figures/step4_final_data_preprocessing")
    figures_output_path.mkdir(parents=True, exist_ok=True)
    acs_metadata_df = pd.read_csv("data/raw_data_subset_for_this_package/metadata_acs.csv")

    samplename_list = acs_metadata_df["sample_name"].unique()

    clean_spectra_A_dict = {}
    clean_spectra_C_dict = {}
    clean_spectra_B_dict = {}
    
    for samplename in samplename_list:
        if "diw" not in samplename:
            run_list = get_runs_list_from_samplename_and_metadata(samplename=samplename, metadata=acs_metadata_df)

            df_arr_A_cumm = pd.DataFrame() 
            df_arr_C_cumm = pd.DataFrame() 

            for i, run_nr in enumerate(run_list):
                                
                try:
                    f = f"run_21_ACS.{run_nr}"
                    df_arr_A, df_arr_C = get_acs_IOP(os.path.join(acs_dir_path, f))

                    df_arr_A_cumm = concatenate_df(df_arr_A_cumm, df_arr_A)
                    df_arr_C_cumm = concatenate_df(df_arr_C_cumm, df_arr_C)
                    
                except Exception as e:
                    handle_exception_when_run_not_working(run=run_nr, e=e)

            n_cumm_sp_A = df_arr_A_cumm.shape[0]
            if n_cumm_sp_A>0:
                df_arr_A_cumm_clean, figA = apply_outlier_detection_pipeline(df_arr_A_cumm, plot=True) 
                median_A_cumm = df_arr_A_cumm_clean.median(axis=0)
                clean_spectra_A_dict[samplename] = median_A_cumm
                figA.savefig(figures_output_path / f"{samplename}_absorption.png")
                
            n_cumm_sp_C = df_arr_C_cumm.shape[0]
            if n_cumm_sp_C>0:
                df_arr_C_cumm_clean, figC = apply_outlier_detection_pipeline(df_arr_C_cumm, plot=True)
                median_C_cumm = df_arr_C_cumm_clean.median(axis=0)
                clean_spectra_C_dict[samplename] = median_C_cumm
                figC.savefig(figures_output_path / f"{samplename}_beam_attenuation.png")
                
            if n_cumm_sp_A > 0 and n_cumm_sp_C > 0:
                # Extract median spectra and wavelength arrays
                waves_a = df_arr_A_cumm_clean.columns.astype("float")
                waves_c = df_arr_C_cumm_clean.columns.astype("float")

                # Calculate scattering
                scattering_spectrum = calculate_scattering_spectrum(median_A_cumm, waves_a, median_C_cumm, waves_c)
                
                # Store in dictionary (using A wavelengths as the reference grid)
                clean_spectra_B_dict[samplename] = scattering_spectrum
            
            plt.close('all')
                
    col_A = df_arr_A.columns
    clean_spectra_A_df = process_and_save_spectra(
        spectra_dict=clean_spectra_A_dict, columns=col_A, output_path=data_output_path, output_filename="acs_a_valid.csv")

    col_C = df_arr_C.columns
    clean_spectra_C_df = process_and_save_spectra(
        spectra_dict=clean_spectra_C_dict, columns=col_C, output_path=data_output_path, output_filename="acs_c_valid.csv")

    col_B = col_A
    clean_spectra_B_df = process_and_save_spectra(
        spectra_dict=clean_spectra_B_dict, columns=col_B, output_path=data_output_path, output_filename="acs_b_valid.csv")


        
if __name__ == "__main__":
    main()