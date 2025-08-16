import numpy as np
from scipy import signal
import numba as nb


# Numba-accelerated function for the CSD computation
@nb.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _compute_csd_numba(fft_segments, n_signals, n_segments, n_freqs, scale, fft_scale):
    """Numba-optimized CSD matrix computation"""
    Pxy = np.zeros((n_signals, n_signals, n_freqs), dtype=np.complex128)
    
    for i in nb.prange(n_signals):
        for j in range(n_signals):
            # For each frequency
            for k in range(n_freqs):
                sum_val = 0.0 + 0.0j
                # For each segment
                for s in range(n_segments):
                    sum_val += fft_segments[i, s, k] * np.conj(fft_segments[j, s, k])
                Pxy[i, j, k] = (sum_val / n_segments) * scale * fft_scale[k]
    
    return Pxy


def pairwise_csd(x, fs=1.0, window='hann', nperseg=None, noverlap=None, 
                               nfft=None, detrend='constant', return_onesided=True, 
                               scaling='density', use_numba=True):
    """
    Highly optimized implementation of the cross-spectral density matrix computation.
    
    Parameters
    ----------
    x : ndarray
        Array containing multiple signals in the first dimension
    fs : float, optional
        Sampling frequency of the x time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. Defaults to 'hann'.
    nperseg : int, optional
        Length of each segment. Defaults to None (uses 256).
    noverlap : int, optional
        Number of points to overlap between segments. Defaults to None (uses nperseg//2).
    nfft : int, optional
        Length of the FFT used. Defaults to None (uses nperseg).
    detrend : str or function or False, optional
        Specifies how to detrend each segment. Defaults to 'constant'.
    return_onesided : bool, optional
        If True, return a one-sided spectrum for real data. Defaults to True.
    scaling : {'density', 'spectrum'}, optional
        Selects between computing the cross spectral density ('density') or cross spectrum ('spectrum').
        Defaults to 'density'.
    use_numba : bool, optional
        Whether to use Numba-accelerated computation. Defaults to True.
        
    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxy : ndarray
        Cross spectral density matrix with shape (n_signals, n_signals, n_freqs).
    """
    # Check if input is complex
    is_complex_input = np.iscomplexobj(x)
    
    # Get the number of signals and samples
    n_signals, n_samples = x.shape
    
    # Set default values for nperseg
    if nperseg is None:
        nperseg = min(256, n_samples)
    
    if noverlap is None:
        noverlap = nperseg // 2
        
    if nfft is None:
        nfft = nperseg
    
    # Get the window - use view instead of copy when possible
    if isinstance(window, str) or isinstance(window, tuple):
        win = signal.get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win) != nperseg:
            raise ValueError('window must have length of nperseg')
    
    # Calculate step size
    step = nperseg - noverlap
    
    # Calculate number of segments
    n_segments = max(1, (n_samples - noverlap) // step)
    
    # Preallocate array for all segments of all signals - more memory efficient
    # Using a 3D array format: [signal, segment, sample]
    all_segments = np.zeros((n_signals, n_segments, nperseg), dtype=x.dtype)
    
    # Create segments efficiently using strided views instead of copies when possible
    for i in range(n_signals):
        s = x[i]
        if n_segments == 1 and len(s) < nperseg:
            # Handle edge case for short signals
            all_segments[i, 0, -len(s):] = s
        else:
            # Create all segments at once
            for j in range(n_segments):
                start = j * step
                end = start + nperseg
                if end <= n_samples:
                    # Use direct indexing to avoid copies
                    all_segments[i, j] = s[start:end]
                else:
                    # Handle partial segment at the end
                    all_segments[i, j, :n_samples-start] = s[start:]
    
    # Apply detrend efficiently
    if detrend == 'constant':
        # Vectorized constant detrend across all segments at once
        segment_means = np.mean(all_segments, axis=2, keepdims=True)
        all_segments -= segment_means
    elif detrend == 'linear':
        # Optimized linear detrend
        t = np.arange(nperseg, dtype=np.float64)
        t_mean = np.mean(t)
        t_centered = t - t_mean
        t_centered_squared_sum = np.sum(t_centered**2)
        
        # Calculate segment means
        segment_means = np.mean(all_segments, axis=2, keepdims=True)
        
        # Calculate slopes efficiently
        for i in range(n_signals):
            for j in range(n_segments):
                segment = all_segments[i, j]
                # Calculate slope for this segment
                slope = np.sum(t_centered * segment) / t_centered_squared_sum
                # Apply linear detrend directly
                all_segments[i, j] = segment - (slope * t + segment_means[i, j, 0] - slope * t_mean)
    elif callable(detrend):
        # Apply custom detrend function
        for i in range(n_signals):
            for j in range(n_segments):
                all_segments[i, j] = detrend(all_segments[i, j])
    
    # Apply window (in-place operation to save memory)
    all_segments *= win

    # Choose the most efficient FFT method based on data
    if np.isrealobj(all_segments):
        # For real data, use rfft which is faster than fft
        if return_onesided:
            fft_segments = np.fft.rfft(all_segments, n=nfft, axis=2)
            freq = np.fft.rfftfreq(nfft, 1/fs)
        else:
            # If we need full spectrum for real data
            fft_segments = np.fft.fft(all_segments, n=nfft, axis=2)
            freq = np.fft.fftfreq(nfft, 1/fs)
            if return_onesided:
                # Keep only positive frequencies
                n_freqs = (nfft // 2) + 1 if nfft % 2 == 0 else (nfft + 1) // 2
                fft_segments = fft_segments[:, :, :n_freqs]
                freq = freq[:n_freqs]
    else:
        # For complex data, always use fft
        fft_segments = np.fft.fft(all_segments, n=nfft, axis=2)
        freq = np.fft.fftfreq(nfft, 1/fs)
        if return_onesided and not is_complex_input:
            # Keep only positive frequencies for real input data
            n_freqs = (nfft // 2) + 1 if nfft % 2 == 0 else (nfft + 1) // 2
            fft_segments = fft_segments[:, :, :n_freqs]
            freq = freq[:n_freqs]
    
    # Get frequency count for final result
    n_freqs = len(freq)
    
    # Calculate scaling factors
    scale = 1.0
    if scaling == 'density':
        scale = 1.0 / (fs * np.sum(win**2))
    elif scaling == 'spectrum':
        scale = 1.0 / (np.sum(win)**2)
    
    # Prepare scaling for one-sided spectrum
    if return_onesided and not is_complex_input:
        if nfft % 2 == 0:  # even nfft
            fft_scale = np.ones(n_freqs)
            fft_scale[1:-1] = 2  # Double everything except DC and Nyquist
        else:  # odd nfft
            fft_scale = np.ones(n_freqs)
            fft_scale[1:] = 2  # Double everything except DC
    else:
        fft_scale = np.ones(n_freqs)
    
    # Choose computation method based on preference and availability
    if use_numba:
        try:
            # Try to use Numba-accelerated function
            Pxy = _compute_csd_numba(fft_segments, n_signals, n_segments, n_freqs, scale, fft_scale)
        except:
            # Fall back to NumPy if Numba fails
            use_numba = False
            
    if not use_numba:
        # Reshape for broadcasting:
        # signal i: (n_signals, 1, n_segments, n_freqs)
        # signal j: (1, n_signals, n_segments, n_freqs)
        fft_i = fft_segments.reshape(n_signals, 1, n_segments, n_freqs)
        fft_j = fft_segments.reshape(1, n_signals, n_segments, n_freqs)
        
        # Compute outer product and mean across segments dimension
        # Result shape: (n_signals, n_signals, n_freqs)
        Pxy = np.mean(fft_i * np.conjugate(fft_j), axis=2)
        
        # Apply scaling factors
        Pxy *= scale * fft_scale
    
    return freq, Pxy
