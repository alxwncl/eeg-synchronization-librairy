import mne

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mne")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

from utils import remove_duplicates, remove_bad_channels, remove_empty_channels

from . import synchronization_csd, synchronization_hilbert



import numpy as np
def differential_window(y, w=100000):
    D = np.gradient(y, axis=1)
    return np.exp(np.abs(D) / w)


def synchronization(file_path, lbands, hbands, window_length, window_overlap, sfreq=256,
                    phase_method="csd", sync_method="wpli", samples_per_epoch=None, epochs_overlap=None,
                    use_numba=True, uncache_data=True, n_jobs=None, verbose=True):
    
    # Handle bad channels
    raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel='auto', verbose=False)
    remove_duplicates(raw)
    remove_bad_channels(raw)
    remove_empty_channels(raw)

    # Get data
    record, _ = raw[:]
    # record = differential_window(record)
    record = record.T
    
    # Compute the syncronization
    if phase_method == "csd":
        if samples_per_epoch is None:
            raise ValueError("samples_per_epoch must be provided for csd method.")
        if epochs_overlap is None:
            raise ValueError("epochs_overlap must be provided for csd method.")
        return synchronization_csd.synchronization(
                record, lbands, hbands, window_length, window_overlap,
                samples_per_epoch, epochs_overlap,  # Specific to csd     
                sfreq=sfreq, method=sync_method, uncache_data=uncache_data, use_numba=use_numba, verbose=verbose
            )
    elif phase_method == "hilbert":
        return synchronization_hilbert.synchronization(
                record, lbands, hbands, window_length, window_overlap,
                n_jobs=n_jobs,  # Specific to hilbert
                sfreq=sfreq, method=sync_method, uncache_data=uncache_data, verbose=verbose
            )
    else:
        raise ValueError(f"Unknown phase_method: {phase_method}. Use 'csd' or 'hilbert'.")
    

__all__ = ["synchronization"]
