import gc
import mne
import numpy as np
import numba as nb
from scipy.signal import hilbert, get_window
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed



@nb.jit(nopython=True, fastmath=True, cache=True)
def _wpli(ref, comp):
    """Numba-optimized WPLI calculation"""
    difference = np.sin(ref - comp)
    denominator = np.sum(np.abs(difference))
    if denominator == 0:
        return 0.0
    return np.abs(np.sum(difference)) / denominator


@nb.jit(nopython=True, fastmath=True, cache=True)
def _pli(ref, comp):
    """Numba-optimized PLI calculation"""
    difference = np.sign(ref - comp)
    return np.abs(np.sum(difference)) / len(difference)


@nb.jit(nopython=True, fastmath=True, cache=True)
def _synchronization_matrix(phases, n_channels, method_flag):
    """
    Ultra-fast synchronization matrix computation with numba
    method_flag: 0 for WPLI, 1 for PLI
    """
    sync_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            if method_flag == 0:
                val = _wpli(phases[:, i], phases[:, j])
            else:
                val = _pli(phases[:, i], phases[:, j])
            
            sync_matrix[i, j] = val
            sync_matrix[j, i] = val
    
    return sync_matrix


def _process_frequency_band(band_index, data, lbands, hbands, window_length, 
                          window_overlap, sfreq, method, n_channels, n_windows, 
                          window_func, window_name):
    """
    Process a single frequency band - designed to be called by multiprocessing
    """
    # Filter data for this band
    band_data = mne.filter.filter_data(
        data.T, sfreq=sfreq, l_freq=lbands[band_index], h_freq=hbands[band_index],
        method='fir', copy=True, verbose=False
    ).T  # Transpose back to (n_samples, n_channels)
    
    # Pre-allocate output for this band
    band_output = np.zeros((n_windows, n_channels, n_channels))
    
    # Compute step size
    step_size = int(window_length * (1 - window_overlap))
    
    # Process windows for this band
    for w in range(n_windows):
        start_idx = w * step_size
        end_idx = start_idx + window_length
        
        # Extract window data
        window_data = band_data[start_idx:end_idx]
        
        # Apply windowing function to reduce spectral leakage
        if window_func is not None:
            # Apply window to each channel
            windowed_data = window_data * window_func[:, np.newaxis]
        else:
            windowed_data = window_data
        
        # Calculate phase for all channels at once using Hilbert transform
        analytic_signal = hilbert(windowed_data, axis=0)
        phases = np.angle(analytic_signal, deg=False)
        
        # Efficiently calculate synchronization matrix
        if method == "wpli":
            band_output[w] = _synchronization_matrix(phases, n_channels, 0)
        elif method == "pli":
            band_output[w] = _synchronization_matrix(phases, n_channels, 1)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Clean up
    del band_data
    gc.collect()
    
    return band_index, band_output


def synchronization(data, lbands, hbands, window_length, window_overlap, 
                    sfreq=256, method="wpli", window_name="hann", uncache_data=False, 
                    verbose=True, n_jobs=None):
    """
    Parallelized implementation using joblib with windowing support
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_channels)
        EEG data
    lbands : list
        Lower frequency bounds for each band
    hbands : list  
        Higher frequency bounds for each band
    window_length : int
        Length of each window in samples
    window_overlap : float
        Overlap between windows (0-1)
    sfreq : float
        Sampling frequency
    method : str
        Synchronization method ('wpli' or 'pli')
    window_name : str or None
        Window function name ('hann', 'hamming', 'blackman', 'bartlett', etc.)
        If None, no windowing is applied
    uncache_data : bool
        Whether to clear data from memory after processing
    verbose : bool
        Whether to show progress bars
    n_jobs : int or None
        Number of parallel jobs (-1 for all cores)
    
    Returns:
    --------
    output : array, shape (n_bands, n_windows, n_channels, n_channels)
        Synchronization matrices for each band and window
    """
    
    # Extract shapes
    n_samples, n_channels = data.shape
    n_bands = len(hbands)
    
    # Compute the number of windows
    step_size = int(window_length * (1 - window_overlap))
    n_windows = (n_samples - window_length) // step_size + 1
    
    # Create window function if specified
    if window_name is not None:
        try:
            window_func = get_window(window_name, window_length)
            if verbose:
                print(f"Using {window_name} window for spectral analysis")
        except ValueError as e:
            print(f"Warning: Invalid window name '{window_name}'. Using no window.")
            print(f"Available windows: hann, hamming, blackman, bartlett, kaiser, etc.")
            window_func = None
    else:
        window_func = None
        if verbose:
            print("No windowing applied")
    
    # Determine number of jobs
    if n_jobs is None:
        n_jobs = -1
    
    # Process bands in parallel
    with tqdm_joblib(total=n_bands, desc=f"Processing {method} for bands", disable=not verbose):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_frequency_band)(
                b, data, lbands, hbands, window_length, window_overlap,
                sfreq, method, n_channels, n_windows, window_func, window_name
            ) for b in range(n_bands)
        )
    
    # Pre-allocate output array
    output = np.zeros((n_bands, n_windows, n_channels, n_channels))
    
    # Collect results in correct order
    for band_index, band_result in results:
        output[band_index] = band_result
    
    # Clear the cache to free up memory
    if uncache_data:
        del data
        gc.collect()
    
    return output
