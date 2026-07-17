from custom_python_tools.acs_data_reader import get_acs_IOP
from custom_python_tools.acs_outlier_detection_pipeline import apply_outlier_detection_pipeline
from custom_python_tools.acs_utils import handle_exception_when_run_not_working
import numpy as np
import pandas as pd
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

### Utils for this script
def analyze_clean_spectra(clean_sp, samplename):
    mean_sp, std_sp = np.mean(clean_sp, axis=0), np.std(clean_sp, axis=0)
    n_valid_sp = len(clean_sp)
    mean_mean = np.mean(mean_sp)
    mean_std = np.mean(std_sp)
    
    analysis_dict = {
            "sample": samplename,
            "n_valid_spectra": n_valid_sp,
            "mean_mean": mean_mean, 
            "mean_std": mean_std,
            "rel_std": 100*mean_std/mean_mean
        }
    
    visual_inspection_needed = _test_if_visual_inspection_is_needed(analysis_dict)
    
    analysis_dict["visual_inspection_needed"] = visual_inspection_needed
    
    return analysis_dict

def _test_if_visual_inspection_is_needed(analysis_dict):
    n = analysis_dict["n_valid_spectra"]
    rel_std = analysis_dict["rel_std"]
    mean_std = analysis_dict["mean_std"] # Ensure this matches your dict key

    # Condition 1: Sample size is too small
    if n < 20:
        return True
    
    # Condition 2: High relative noise AND significant absolute noise
    if n >= 20:
        if rel_std > 5 and mean_std > 1:
            return True
    
    # Otherwise, it's considered a valid/clean sample
    return False

def savefig_in_correct_folder(output_path, fig, samplename, f, analysis_dict, sampletype):
    visual_inspection_needed = analysis_dict["visual_inspection_needed"]
    
    if visual_inspection_needed:
        output_dir = Path(f"{output_path}/visual inspection needed")
    else: 
        output_dir = Path(f"{output_path}/valid")

    # Create the directory (and any missing parent folders)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Now save your plot
    fig.savefig(output_dir / f"{f}_{samplename}_{sampletype}_relSTD={analysis_dict['rel_std']:.1f}_STD={analysis_dict['mean_std']:.1f}.png")

def plot_outlier_statistics(df_outlier_report_A, df_outlier_report_S):
    # Create the figure and axes
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))

    # --- Row 1: Absorption Data (df_outlier_report_A) ---

    # Left: Mean Std vs N Valid Spectra
    sns.scatterplot(data=df_outlier_report_A, x="n_valid_spectra", y="mean_std", 
                    ax=ax[0, 0], color='tab:blue', s=60)
    ax[0, 0].set_title("Absorption: Std vs. number of valid spectra", fontsize=14)
    ax[0, 0].set_xlabel("Number of valid spectra")
    ax[0, 0].set_ylabel("Mean Standard Deviation [1/m]")

    # Right: Relative Std vs N Valid Spectra
    sns.scatterplot(data=df_outlier_report_A, x="n_valid_spectra", y="rel_std", 
                    ax=ax[0, 1], color='tab:blue', s=60)
    ax[0, 1].set_title("Absorption: rel Std vs. number of valid spectra", fontsize=14)
    ax[0, 1].set_xlabel("Number of Valid Spectra")
    ax[0, 1].set_ylabel("Relative Std [%]")


    # --- Row 2: Scattering Data (df_outlier_report_S) ---

    # Left: Mean Std vs N Valid Spectra
    sns.scatterplot(data=df_outlier_report_S, x="n_valid_spectra", y="mean_std", 
                    ax=ax[1, 0], color='tab:orange', s=60)
    ax[1, 0].set_title(r"Scattering: Std vs. number of valid spectra", fontsize=14)
    ax[1, 0].set_xlabel("Number of valid spectra")
    ax[1, 0].set_ylabel("Mean Standard Deviation [1/m]")

    # Right: Relative Std vs N Valid Spectra
    sns.scatterplot(data=df_outlier_report_S, x="n_valid_spectra", y="rel_std", 
                    ax=ax[1, 1], color='tab:orange', s=60)
    ax[1, 1].set_title("Scattering: rel Std vs. number of valid spectra", fontsize=14)
    ax[1, 1].set_xlabel("Number of valid spectra")
    ax[1, 1].set_ylabel("Relative Std [%]")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    return fig

### main script

def main():

    dir_path = "data/raw_data_subset_for_this_package/1_acs_runs"
    output_path = Path("figures/step2_identification_outlier_runs_results")
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_list = [ff for ff in os.listdir(dir_path) if ff.startswith('run_21_ACS.')]

    metadata = pd.read_csv("data/raw_data_subset_for_this_package/metadata_acs.csv")

    saved_data_a = []
    saved_data_s = []

    for f in file_list:
        try:
            
            samplename = metadata.loc[metadata["run_nr"]==int(f[-3:]), "sample_name"].values[0]
            
            if "diw" not in samplename:
            
                df_arr_A, df_arr_S = get_acs_IOP(os.path.join(dir_path, f))
            
                # 1. Generate the figures (they are stored in Matplotlib's internal memory)
                clean_A, fig_A = apply_outlier_detection_pipeline(df_arr_A, plot=True)
                analyzis_clean_A = analyze_clean_spectra(clean_A, samplename)
                saved_data_a.append(analyzis_clean_A)
                
                savefig_in_correct_folder(output_path, fig_A, samplename, f, analyzis_clean_A, sampletype="A")
                        
                clean_S, fig_S = apply_outlier_detection_pipeline(df_arr_S, plot=True)
                analyzis_clean_S = analyze_clean_spectra(clean_S, samplename)
                saved_data_s.append(analyzis_clean_S)

                savefig_in_correct_folder(output_path, fig_S, samplename, f, analyzis_clean_S, sampletype="S")
                
                plt.close('all')
            
        except Exception as e:
            handle_exception_when_run_not_working(e=e)


    df_outlier_report_A = pd.DataFrame(saved_data_a)
    df_outlier_report_S = pd.DataFrame(saved_data_s)

    fig = plot_outlier_statistics(df_outlier_report_A, df_outlier_report_S)
    fig.savefig(f"{output_path}/spectra_mapping.png")
    
    plt.close("all")
    
    df_outlier_report_A.to_csv(f"{output_path}/outlier_report_a.csv")
    df_outlier_report_S.to_csv(f"{output_path}/outlier_report_s.csv")
    
### running the main script

if __name__ == "__main__":
    main()