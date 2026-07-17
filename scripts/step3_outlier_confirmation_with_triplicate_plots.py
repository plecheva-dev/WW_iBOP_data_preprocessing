from custom_python_tools.acs_data_reader import get_acs_IOP
from custom_python_tools.acs_outlier_detection_pipeline import apply_outlier_detection_pipeline
from custom_python_tools.acs_utils import concatenate_df, convert_run_list_to_string, get_runs_list_from_samplename_and_metadata
from custom_python_tools.acs_utils import handle_exception_when_run_not_working
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

### Utils for this script ###
def plot_median_and_iqr(ax, wav, spectra_df, label, color, ls="-"):
   
    median_sp = spectra_df.median(axis=0)
    p25 = spectra_df.quantile(0.25, axis=0)
    p75 = spectra_df.quantile(0.75, axis=0)
    
    # Plot Median line
    ax.plot(wav, median_sp, color=color, lw=2, label=f"Median {label}(n={len(spectra_df)})", ls=ls)
    
    # Plot Shaded IQR area
    ax.fill_between(wav, p25, p75, color=color, alpha=0.2)
    
    ax.legend(loc='upper right', fontsize='small')

def get_plot_style(run, outliers, dubious):
    """Returns line style, label status, and color based on run classification."""
    if run in outliers:
        return "-.", f"{run} (outlier)", "red"
    if run in dubious:
        return "--", f"{run} (dubious)", "orange"
    return "-", f"{run} (normal)", "green"

def apply_outlier_detection_pipeline_and_plot(df_arr, wav, ax, label, color="black", ls="-"):
    df_arr_clean, _ = apply_outlier_detection_pipeline(df_arr, plot=False)
    n_valid_spectra = df_arr_clean.shape[0]
    if n_valid_spectra > 0:
        plot_median_and_iqr(
                ax=ax, wav=wav, spectra_df=df_arr_clean, label=label, color=color, ls=ls)


### main function for this script ###


def main():
    
    acs_dir_path = "data/raw_data_subset_for_this_package/1_acs_runs"

    output_dir_path = Path("figures/step3_outlier_confirmation_with_triplicate_plots")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv("data/raw_data_subset_for_this_package/metadata_acs.csv")
    samplename_list = metadata_df["sample_name"].unique()

    visual_outlier_runs_A_list = [26, 448, 463, 465, 473, 474, 481]
    dubious_runs_A_list = [472, 58, 394, 439, 441]
    visual_outlier_runs_C_list = [394, 413, 414, 463]
    dubious_runs_C_list = [26, 44, 67, 441]


    outlier_A = convert_run_list_to_string(visual_outlier_runs_A_list)
    outlier_C = convert_run_list_to_string(visual_outlier_runs_C_list)
    dubious_A = convert_run_list_to_string(dubious_runs_A_list)
    dubious_C = convert_run_list_to_string(dubious_runs_C_list)
    
    for samplename in samplename_list:
        if "diw" not in samplename:
            run_list = get_runs_list_from_samplename_and_metadata(samplename=samplename, metadata=metadata_df)

            fig, ax = plt.subplots(ncols=2)

            df_arr_A_cumm = pd.DataFrame() 
            df_arr_C_cumm = pd.DataFrame() 

            for i, run_nr in enumerate(run_list):
                
                ls_A, label_A, color_A = get_plot_style(run_nr, outlier_A, dubious_A)
                ls_C, label_C, color_C = get_plot_style(run_nr, outlier_C, dubious_C)
                    
                    
                try:
                    acs_filename = f"run_21_ACS.{run_nr}"
                    df_arr_A, df_arr_C = get_acs_IOP(os.path.join(acs_dir_path, acs_filename))
                    
                    wav_A = np.array(df_arr_A.columns.astype(float))
                    df_arr_A_cumm = concatenate_df(df_arr_A_cumm, df_arr_A)
                    apply_outlier_detection_pipeline_and_plot(df_arr=df_arr_A, wav=wav_A, ax=ax[0], label=label_A, color=color_A, ls=ls_A)
                    
                    wav_C = np.array(df_arr_C.columns.astype(float))
                    df_arr_C_cumm = concatenate_df(df_arr_C_cumm, df_arr_C)
                    apply_outlier_detection_pipeline_and_plot(df_arr=df_arr_C, wav=wav_C, ax=ax[1], label=label_C, color=color_C, ls=ls_C)
                    
                except Exception as e:
                    handle_exception_when_run_not_working(e=e)
            
            try: 
                apply_outlier_detection_pipeline_and_plot(df_arr=df_arr_A_cumm, wav=wav_A, ax=ax[0], label="(combined)")
            except: 
                print(f"Could not plot data for the absorption spectra of {samplename}")
            
            try:
                apply_outlier_detection_pipeline_and_plot(df_arr=df_arr_C_cumm, wav=wav_C, ax=ax[1], label="(combined)")     
            except: 
                print(f"Could not plot data for the beam attenuation spectra of {samplename}")

            ax[0].legend()
            ax[1].legend()

            plt.title(samplename)
            plt.savefig(output_dir_path / f"{samplename}.png")
            plt.close()

if __name__ == "__main__":
    main()