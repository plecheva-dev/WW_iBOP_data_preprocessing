from custom_python_tools.acs_data_reader import get_acs_IOP
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

dir_path = "data/raw_data_subset_for_this_package/1_acs_runs"
metadata = pd.read_csv("data/raw_data_subset_for_this_package/metadata_acs.csv")

# Get all relevant run files
file_list = [ff for ff in os.listdir(dir_path) if ff.startswith('run_21_ACS.')]

if __name__ == "__main__":
    results = []

    for f in file_list:
        # Extract run number from filename (assuming 'run_21_ACS.XXX' format)
        run_nr = int(f.split('.')[-1])
        
        try:
            # Load the data
            df_arr_A, df_arr_C = get_acs_IOP(os.path.join(dir_path, f))
            
            # Record the counts
            results.append({
                "run_nr": run_nr,
                "count_A": df_arr_A.shape[0],
                "count_C": df_arr_C.shape[0]
            })
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # Create a DataFrame for easy plotting
    df_counts = pd.DataFrame(results).sort_values("run_nr")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    plt.scatter(df_counts["run_nr"], df_counts["count_A"], marker='o', label='Absorption (A)')
    plt.scatter(df_counts["run_nr"], df_counts["count_C"], marker='s', label='Attenuation (C)')

    plt.title("Number of Spectra per ACS Run", fontsize=14)
    plt.xlabel("Run Number", fontsize=12)
    plt.ylabel("Count (Number of Spectra)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.show()