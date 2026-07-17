import pandas as pd


def concatenate_df(df_cumm: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Appends a DataFrame to a cumulative DataFrame, initializing it if empty.

    Args:
        df_cumm (pd.DataFrame): The running, cumulative DataFrame. Can be None 
            or empty.
        df (pd.DataFrame): The new DataFrame containing data to append.

    Returns:
        pd.DataFrame: A combined DataFrame containing the data from both inputs.
    """
    # Check if df_cumm is None or empty
    if df_cumm is None or df_cumm.empty:
        df_cumm = df.copy()
    else:
        df_cumm = pd.concat([df_cumm, df], ignore_index=True)
    
    return df_cumm

def convert_run_list_to_string(run_list):
    for (i, r) in enumerate(run_list):
        if r<10:
            run_list[i] = f"00{r}"
        elif r<100:
            run_list[i] = f"0{r}"
        else:
            run_list[i] = str(r)
    # Return as a list
    return run_list

def get_runs_list_from_samplename_and_metadata(samplename, metadata):
    # Filter rows where sample_name matches, then grab the run_nr column
    runs = metadata.loc[metadata["sample_name"] == samplename, "run_nr"].tolist()
    runs = convert_run_list_to_string(runs)
    return runs

def handle_exception_when_run_not_working(run, e):
    print(f"FAILED FOR RUN: {run}")
    print(f"ERROR TYPE: {type(e).__name__}")
    print(f"ERROR MESSAGE: {e}")
    
