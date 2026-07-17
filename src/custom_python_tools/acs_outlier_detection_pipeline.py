import numpy as np
import matplotlib.pyplot as plt
from .acs_outlier_detection_functions.iterative import apply_asymmetric_outlier_detection
from .acs_outlier_detection_functions.minmax import apply_minmax_outlier_detection_and_correction
from .acs_outlier_detection_functions.slope_576nm import apply_slope576_outlier_detection_and_correction



def subplot_helper(ax, wav, spectra_df, title, display_stats=False):
    """A helper function to plot the spectra contained in a pandas DataFrame (spectra_df) on the figure ax.

    Args:
        ax (matplotlib): the figure ax on which to plot the data
        wav (array): the wavelengths of the spectra to plot
        spectra_df (pd.DataFrame): the spectra to plot
        title (string): the title of the ax.
        display_stats (bool, optional): True if you want to add the median and IQR spectra to the ax. Defaults to False.
    """
    # Plot individual spectra
    for _, sp in spectra_df.iterrows():
        ax.plot(wav, sp, color="gray", alpha=0.3, lw=0.5)
    
    if display_stats:
        # Calculate Median and IQR (25th and 75th percentiles)
        # Using numeric_only=True if your DF has non-numeric columns
        median_sp = spectra_df.median(axis=0)
        p25 = spectra_df.quantile(0.25, axis=0)
        p75 = spectra_df.quantile(0.75, axis=0)
        
        # Plot Median line
        ax.plot(wav, median_sp, color="blue", lw=2, label="Median")
        
        # Plot Shaded IQR area
        ax.fill_between(wav, p25, p75, color="blue", alpha=0.2, label="IQR (25-75%)")
        
        ax.legend(loc='upper right', fontsize='small')
    
    # set the ax title
    n_sp = spectra_df.shape[0]
    ax.set_title(f"{title} (n= {n_sp})")


def apply_outlier_detection_pipeline(spectra_df, plot=True):
    """A four step pipeline to detect outliers in a dataframe of spectra. The methods are tailored to the ACs data.

    Args:
        spectra_df (pd.DataFrame): the spectra to apply the pipeline to.
        plot (bool, optional): Generate a summary plot of the pipeline if True. Defaults to True.

    Returns:
        valid_spectra_df (pd.DataFrame): a DaaFrame containing only the valid spectra.
        fig (matplotlip): if plot=True, returns a summary plot. Else, returns None.
    """
    
    spectra_df = spectra_df.reset_index(drop=True)

    # step 1
    minmax_outlier_index, minmax_valid_index, updated_df = apply_minmax_outlier_detection_and_correction(spectra_df)
    remaining_spectra_after_minmax_df = updated_df.loc[minmax_valid_index]
    
    # step 2, applied only to wavelengths below 576 nm
    wav = np.array(spectra_df.columns.astype("float"))
    mask = wav < 576
    
    iterative_outlier_index_1, iter1_valid_index = apply_asymmetric_outlier_detection(remaining_spectra_after_minmax_df.loc[:, mask], threshold_iter=5, mad_floor_perc=0.2)
    remaining_spectra_after_first_iterative_df = updated_df.loc[iter1_valid_index]
    
    # step 3          
    slope576_outlier_index, valid_index, updated_df = apply_slope576_outlier_detection_and_correction(remaining_spectra_after_first_iterative_df)       
    remaining_spectra_after_slope576_df = updated_df.loc[valid_index]
    
    # step 4
    iterative_outlier_index_2, valid_index = apply_asymmetric_outlier_detection(remaining_spectra_after_slope576_df, threshold_iter=2)
    valid_spectra_df = updated_df.loc[valid_index]
    
    # generating a summary plot if plot = True
    fig = None
    
    if plot: 
        fig, ax = plt.subplots(nrows=4, ncols=3, figsize = (15, 10), sharex=True)
        
        all_spectra = spectra_df
        minmax_outliers = spectra_df.loc[minmax_outlier_index]
        remaining_after_minmax = remaining_spectra_after_minmax_df
        iter1_outlier = remaining_spectra_after_minmax_df.loc[iterative_outlier_index_1]
        remaining_after_iter1 = remaining_spectra_after_first_iterative_df
        slope576_outlier = remaining_spectra_after_first_iterative_df.loc[slope576_outlier_index]
        remaining_after_slope = remaining_spectra_after_slope576_df
        iter2_outlier = remaining_spectra_after_slope576_df.loc[iterative_outlier_index_2]
        
        subplot_helper(
            ax=ax[0, 0], wav=wav, spectra_df=all_spectra, title="All spectra", display_stats=True)
        ax[0, 0].plot(wav, valid_spectra_df.median(axis=0), ls="--", label="Final valid median")
        subplot_helper(
            ax=ax[1, 0], wav=wav, spectra_df=minmax_outliers, title="MinMax outlier")
        
        subplot_helper(
            ax=ax[2, 0], wav=wav, spectra_df=remaining_after_minmax, title="Remaining after MiMmax", display_stats=True)
        subplot_helper(
            ax=ax[3, 0], wav=wav, spectra_df=iter1_outlier, title="Iter1 outlier")
        
        subplot_helper(
            ax=ax[0, 1], wav=wav, spectra_df=remaining_after_iter1, title="Remaining after Iter1", display_stats=True)
        subplot_helper(
            ax=ax[1, 1], wav=wav, spectra_df=slope576_outlier, title="Slope outliers")
        
        subplot_helper(
            ax=ax[2, 1], wav=wav, spectra_df=remaining_after_slope, title="Remaining after Slope", display_stats=True)
        subplot_helper(
            ax=ax[3, 1], wav=wav, spectra_df=iter2_outlier, title="Iter2 outlier")

        subplot_helper(
            ax=ax[3, 2], wav=wav, spectra_df=valid_spectra_df, title="Valid spectra", display_stats=True)
        
        plt.tight_layout()

    return valid_spectra_df, fig

