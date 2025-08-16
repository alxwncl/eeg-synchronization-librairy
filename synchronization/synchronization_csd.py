import gc
import mne
import math
import numpy as np
import numba as nb
from tqdm import tqdm

from .signal_analysis import pairwise_csd


def _epoch_data(data, epoch_length, epoch_overlap, uncache_data=False):
    """
    Segment raw data into sliding time epochs.
    
    Parameters:
        data (np.ndarray):       Array of raw data with shape (n_samples, n_channels).
        epoch_length (int):      Number of samples within each epoch.
        epoch_overlap (float):   Fraction of overlap between consecutive epochs (value between 0 and 1).
    
    Returns:
        np.ndarray: epoched data with shape (n_epochs, epoch_length, n_channels).
    """
    if not 0 <= epoch_overlap < 1:
        raise ValueError("epoch overlap must be between 0 and 1")
    
    n_samples, n_channels = data.shape
    step_size = int(epoch_length * (1 - epoch_overlap))
    
    if step_size == 0:
        raise ValueError("Overlap too high, resulting in zero step size")
    
    # Calculate number of epochs
    n_epochs = (n_samples - epoch_length) // step_size + 1
    
    if n_epochs <= 0:
        return np.empty((0, epoch_length, n_channels))
    
    # Pre-allocate output array
    epochs = np.zeros((n_epochs, epoch_length, n_channels))
    
    # Fill the epochs using vectorized operations
    for i in range(n_epochs):
        start_idx = i * step_size
        end_idx = start_idx + epoch_length
        epochs[i] = data[start_idx:end_idx]

    # Clear the cache to free up memory
    if uncache_data:
        del data
        gc.collect()
    
    return epochs


def _epochs_csd(data, sfreq, use_numba=True, uncache_data=False, verbose=True):
    """
    Compute the imaginary part of pairwise csd.

    Returns:
        csds : np.ndarray, shape (n_epochs, n_f_bins, n_channels, n_channels)
        f_bins: np.ndarray, frequency bin centers
    """
    n_epochs, n_samples, n_channels = data.shape

    # STFT parameters (same as your previous nperseg/noverlap/nfft)
    window  = 'hamming'
    nperseg = (2 * n_samples) // 9
    noverlap= nperseg // 2
    nfft    = max(256, 2**math.ceil(math.log2(nperseg)))
    n_freqs = nfft // 2 + 1

    # Preallocate the full result
    csds = np.zeros((n_epochs, n_freqs, n_channels, n_channels), dtype=float)

    # Loop over epochs
    for e in tqdm(range(n_epochs), desc='Computing csd for epochs', disable=not verbose):

        # Apply pairwise CSD for the epoch
        f_bins, csd_epoch = pairwise_csd(
            data[e].T,
            fs=sfreq,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            return_onesided=True,
            scaling='density',
            use_numba=use_numba
        )

        # Only keep the imaginary part
        csds[e] = np.imag(csd_epoch.T)

    # Clear the cache to free up memory
    if uncache_data:
        del data
        gc.collect()

    return csds, f_bins


def _pli_matrix(window_data, n_channels, n_bands, band_masks):
    """
    Compute PLI matrices for all channel pairs.
    
    Parameters:
        window_data : np.ndarray, shape (n_epochs, n_freqs, n_channels, n_channels)
                      imaginary part of cross-spectral density
        n_channels  : int, number of channels
        n_bands     : int, number of frequency bands
        band_masks  : list of boolean arrays, frequency band masks
        
    Returns:
        result : np.ndarray, shape (n_bands, n_channels, n_channels)
                 PLI values for each band and channel pair
    """
    import numpy as np
    
    result = np.zeros((n_bands, n_channels, n_channels))

    # Get upper triangle indices
    i_indices, j_indices = np.triu_indices(n_channels, k=1)
    
    # Vectorized computation for all channel pairs at once
    for i_idx, j_idx in zip(i_indices, j_indices):
        # Extract CSD values for this channel pair
        csd_pair = window_data[:, :, i_idx, j_idx]
        
        # Compute PLI
        pli_all_freq = np.abs(np.mean(np.sign(csd_pair), axis=0))
        
        # Replace NaN values with 0
        pli_all_freq = np.nan_to_num(pli_all_freq)
        
        # Process each frequency band
        for b, mask in enumerate(band_masks):
            if np.any(mask):
                band_pli = np.mean(pli_all_freq[mask])
            else:
                band_pli = 0
            
            # Fill symmetric matrix
            result[b, i_idx, j_idx] = band_pli
            result[b, j_idx, i_idx] = band_pli

    return result


@nb.njit(parallel=False, fastmath=True, cache=True)
def _pli_matrix_numba(window_data, n_channels, n_bands, band_masks):
    """
    Compute PLI matrices for all channel pairs using Numba for acceleration.
    
    Parameters:
        window_data : np.ndarray, shape (n_epochs, n_freqs, n_channels, n_channels)
                      imaginary part of cross-spectral density
        n_channels  : int, number of channels
        n_bands     : int, number of frequency bands
        band_masks  : list of boolean arrays, frequency band masks
        
    Returns:
        result : np.ndarray, shape (n_bands, n_channels, n_channels)
                 PLI values for each band and channel pair
    """
    result = np.zeros((n_bands, n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            csd_pair = window_data[:, :, i, j]
            n_epochs, n_freqs = csd_pair.shape
            
            # Compute PLI for all frequencies
            pli_all_freq = np.zeros(n_freqs)
            for f in range(n_freqs):
                # Calculate the sign of imaginary part for all epochs
                sign_sum = 0.0
                for e in range(n_epochs):
                    # Sign of the imaginary part (which is already extracted in window_data)
                    sign_value = 1.0 if csd_pair[e, f] > 0 else (-1.0 if csd_pair[e, f] < 0 else 0.0)
                    sign_sum += sign_value
                
                # Calculate PLI as absolute value of mean sign
                pli_all_freq[f] = abs(sign_sum / n_epochs) if n_epochs > 0 else 0.0
            
            # Process by frequency band
            for b in range(len(band_masks)):
                mask = band_masks[b]
                if np.sum(mask) > 0:
                    # Calculate mean for this band
                    band_sum = 0.0
                    count = 0
                    for f in range(n_freqs):
                        if mask[f]:
                            band_sum += pli_all_freq[f]
                            count += 1
                    
                    band_pli = band_sum / count if count > 0 else 0.0
                else:
                    band_pli = 0.0
                
                # Fill symmetric matrix
                result[b, i, j] = band_pli
                result[b, j, i] = band_pli
                
    return result


def _wpli_matrix(window_data, n_channels, n_bands, band_masks):
    result = np.zeros((n_bands, n_channels, n_channels))

    # Get upper triangle indices
    i_indices, j_indices = np.triu_indices(n_channels, k=1)
    
    # Vectorized computation for all channel pairs at once
    for i_idx, j_idx in zip(i_indices, j_indices):
        # Extract all pairs at once
        csd_pair = window_data[:, :, i_idx, j_idx]
        
        # Compute wPLI
        numerator = np.abs(np.mean(csd_pair, axis=0))
        denominator = np.mean(np.abs(csd_pair), axis=0)
        wpli_all_freq = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        
        # Process each frequency band
        for b, mask in enumerate(band_masks):
            if np.any(mask):
                band_wpli = np.mean(wpli_all_freq[mask])
            else:
                band_wpli = 0
            
            # Fill symmetric matrix
            result[b, i_idx, j_idx] = band_wpli
            result[b, j_idx, i_idx] = band_wpli

    return result


@nb.njit(parallel=False, fastmath=True, cache=True)
def _wpli_matrix_numba(window_data, n_channels, n_bands, band_masks):
    result = np.zeros((n_bands, n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            csd_pair = window_data[:, :, i, j]
            n_epochs, n_freqs = csd_pair.shape
            
            # Manual implementation of mean along axis 0
            wpli_all_freq = np.zeros(n_freqs)
            for f in range(n_freqs):
                # Calculate numerator (abs of mean)
                sum_val = 0.0
                for e in range(n_epochs):
                    sum_val += csd_pair[e, f]
                mean_val = sum_val / n_epochs
                numerator = abs(mean_val)
                
                # Calculate denominator (mean of abs)
                sum_abs = 0.0
                for e in range(n_epochs):
                    sum_abs += abs(csd_pair[e, f])
                denominator = sum_abs / n_epochs
                
                # Calculate wPLI
                if denominator != 0:
                    wpli_all_freq[f] = numerator / denominator
            
            # Process by frequency band
            for b in range(len(band_masks)):
                mask = band_masks[b]
                if np.sum(mask) > 0:
                    # Calculate mean for this band
                    band_sum = 0.0
                    count = 0
                    for f in range(n_freqs):
                        if mask[f]:
                            band_sum += wpli_all_freq[f]
                            count += 1
                    
                    band_wpli = band_sum / count if count > 0 else 0.0
                else:
                    band_wpli = 0.0
                
                # Fill symmetric matrix
                result[b, i, j] = band_wpli
                result[b, j, i] = band_wpli
                
    return result

def synchronization(data, lbands, hbands, epochs_per_window, window_overlap,
                    samples_per_epoch, epochs_overlap, 
                    sfreq=256, method="wpli", use_numba=True, uncache_data=True, verbose=True):

    # Epoch the data
    epoched_data = _epoch_data(data, samples_per_epoch, epochs_overlap, uncache_data=uncache_data)

    # Compute the CSD
    csds, f_bins = _epochs_csd(epoched_data, sfreq=sfreq, verbose=verbose, uncache_data=uncache_data)

    # Extract the shapes
    n_epochs, _, n_channels, _ = csds.shape
    n_bands = len(hbands)
    
    # Calculate number of windows
    stride = epochs_per_window - window_overlap
    n_windows = max(0, 1 + (n_epochs - epochs_per_window) // stride)
    
    if n_windows == 0:
        return np.empty((n_bands, 0, n_channels, n_channels))
    
    # Output array
    output = np.zeros((n_bands, n_windows, n_channels, n_channels))
    
    # Pre-compute band masks
    band_masks = [(f_bins >= lband) & (f_bins <= hband) for lband, hband in zip(lbands, hbands)]
    
    for w in tqdm(range(n_windows), desc=f'Computing {method} for windows', disable=not verbose):
        start_idx = w * stride
        window_data = csds[start_idx:start_idx + epochs_per_window]
        
        # Use JIT-compiled function for the core computation
        if method == "wpli":
            if use_numba:
                output[:, w] = _wpli_matrix_numba(window_data, n_channels, n_bands, band_masks)
            else:
                output[:, w] = _wpli_matrix(window_data, n_channels, n_bands, band_masks)
        elif method == "pli":
            if use_numba:
                output[:, w] = _pli_matrix_numba(window_data, n_channels, n_bands, band_masks)
            else:
                output[:, w] = _pli_matrix(window_data, n_channels, n_bands, band_masks)
        else:
            raise ValueError("Invalid method. Use 'wpli' or 'pli'.")
    
    # Clear the cache to free up memory
    if uncache_data:
        del csds
        gc.collect()

    return output



if __name__ == '__main__':

    # Load the EDF file
    file_path = '..\\..\\Dataset\\chb-mit-scalp-eeg-database-1.0.0\\chb01\\chb01_01.edf'
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    record, times = raw[:]
    record = record.T

    # Parameters
    sfreq = 256
    lbands = [10]
    hbands = [13]
    samples_per_epoch = 1280
    epochs_overlap = 0.5
    epochs_per_window = 9
    shared_epochs = 7

    # Test functions
    wplis = synchronization(record, lbands, hbands, epochs_per_window, shared_epochs, samples_per_epoch, epochs_overlap, sfreq=sfreq, method='wpli', verbose=True)

    wplis.tofile('original.txt', sep=' ', format='%.6f')
