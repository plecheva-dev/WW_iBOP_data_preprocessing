import numpy as np

def apply_asymmetric_outlier_detection(spectra_df, threshold_iter=5.0, max_iter=50, low_percentile=10, mad_floor_perc=0.1):
    data = spectra_df.values 
    n_samples = data.shape[0]
    outlier_mask = np.zeros(n_samples, dtype=bool)

    for _ in range(max_iter):
        valid_indices = np.where(~outlier_mask)[0]
        if len(valid_indices) <= 2: # Changed to 2: need enough for statistics
            break
            
        current = data[valid_indices]
        ref_spectrum = np.percentile(current, q=low_percentile, axis=0)
        mad_floor = max(0.25, mad_floor_perc * ref_spectrum.mean(axis=0))
        
        # Calculate positive-only residuals
        positive_diffs = np.maximum(current - ref_spectrum, 0)
        scores = np.linalg.norm(positive_diffs, axis=1)

        # Robust MAD-based threshold
        med_score = np.median(scores)
        mad_score = np.median(np.abs(scores - med_score))

        # --- THE REFINED LOGIC ---
        # Instead of breaking, we enforce a minimum "spread" (the noise floor).
        # This prevents the threshold from becoming 0, but still allows us 
        # to catch a few stray outliers that are way above the floor.
        effective_mad = max(mad_score, mad_floor)
        
        limit = med_score + (threshold_iter * effective_mad)
        # -------------------------

        worst_local_idx = np.argmax(scores)
        worst_score = scores[worst_local_idx]

        # If the worst spectrum is still within our "Noise Floor Gate", we stop.
        # Otherwise, we keep removing until the "Monsters" are gone.
        if worst_score <= limit:
            break

        outlier_mask[valid_indices[worst_local_idx]] = True

    return spectra_df.index[outlier_mask].values, spectra_df.index[~outlier_mask].values