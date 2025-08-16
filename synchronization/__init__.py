"""
EEG Synchronization Analysis Module

This module provides functionality for computing synchronization measures
from EEG data using different phase extraction methods (CSD and Hilbert)
and various synchronization metrics.
"""

import warnings
from pathlib import Path
from typing import Union, List, Optional, Tuple, Any
from enum import Enum

import mne
import numpy as np

# Suppress specific MNE warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mne")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

try:
    from utils import remove_duplicates, remove_bad_channels, remove_empty_channels
except ImportError:
    from ..utils import remove_duplicates, remove_bad_channels, remove_empty_channels

from . import synchronization_csd, synchronization_hilbert


class PhaseMethod(Enum):
    """Supported phase extraction methods."""
    CSD = "csd"
    HILBERT = "hilbert"


class SyncMethod(Enum):
    """Common synchronization methods."""
    WPLI = "wpli"
    PLI = "pli"


def differential_window(
    data: np.ndarray, 
    window_param: float = 100000.0
) -> np.ndarray:
    """
    Apply differential windowing to EEG data.
    
    This function computes the exponential of the absolute gradient
    normalized by a window parameter, which can be used for data
    preprocessing or feature extraction.
    
    Args:
        data: Input EEG data with shape (channels, samples)
        window_param: Window parameter for normalization (default: 100000.0)
        
    Returns:
        Processed data with same shape as input
        
    Raises:
        ValueError: If data is not 2D or window_param is not positive
    """
    if data.ndim != 2:
        raise ValueError(
            f"Data must be 2D (channels, samples), got shape: {data.shape}"
        )
    
    if window_param <= 0:
        raise ValueError(f"Window parameter must be positive, got: {window_param}")
    
    # Compute gradient along the time axis (axis=1)
    gradient = np.gradient(data, axis=1)
    
    # Apply exponential transformation
    return np.exp(np.abs(gradient) / window_param)


def _load_and_preprocess_eeg(
    file_path: Union[str, Path],
    verbose: bool = True
) -> Tuple[np.ndarray, mne.io.Raw]:
    """
    Load and preprocess EEG data from file.
    
    Args:
        file_path: Path to the EEG file
        verbose: Whether to show loading progress
        
    Returns:
        Tuple of (preprocessed_data, raw_object)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"EEG file not found: {file_path}")
    
    if not file_path.suffix.lower() in ['.edf', '.bdf']:
        raise ValueError(
            f"Unsupported file format: {file_path.suffix}. "
            "Supported formats: .edf, .bdf"
        )
    
    try:
        # Load the EEG data
        raw = mne.io.read_raw_edf(
            str(file_path), 
            preload=True, 
            stim_channel='auto', 
            verbose=verbose
        )
        
        # Apply preprocessing steps
        remove_duplicates(raw)
        remove_bad_channels(raw)
        remove_empty_channels(raw)
        
        # Extract data and transpose to (samples, channels)
        data, _ = raw[:]
        data = data.T
        
        return data, raw
        
    except Exception as e:
        raise ValueError(f"Error loading EEG file {file_path}: {e}")


def _validate_synchronization_params(
    phase_method: PhaseMethod,
    samples_per_epoch: Optional[int],
    epochs_overlap: Optional[Union[int, float]],
    window_length: int,
    window_overlap: Union[int, float],
    lbands: List[float],
    hbands: List[float]
) -> None:
    """
    Validate synchronization parameters.
    
    Args:
        phase_method: Phase extraction method
        samples_per_epoch: Samples per epoch (required for CSD)
        epochs_overlap: Overlap between epochs (required for CSD)
        window_length: Length of processing windows
        window_overlap: Overlap between windows
        lbands: Low frequency bands
        hbands: High frequency bands
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Validate phase method specific parameters
    if phase_method == PhaseMethod.CSD:
        if samples_per_epoch is None:
            raise ValueError(
                "samples_per_epoch must be provided for CSD phase method"
            )
        if epochs_overlap is None:
            raise ValueError(
                "epochs_overlap must be provided for CSD phase method"
            )
        if samples_per_epoch <= 0:
            raise ValueError(
                f"samples_per_epoch must be positive, got: {samples_per_epoch}"
            )
    
    # Validate general parameters
    if window_length <= 0:
        raise ValueError(f"window_length must be positive, got: {window_length}")
    
    if not (0 <= window_overlap < 1 if window_overlap < 1 else 0 <= window_overlap < window_length):
        raise ValueError(
            f"window_overlap must be between 0 and 1 (fraction) or "
            f"0 and window_length (samples), got: {window_overlap}"
        )
    
    if not lbands or not hbands:
        raise ValueError("Both lbands and hbands must be non-empty lists")
    
    if len(lbands) != len(hbands):
        raise ValueError(
            f"lbands and hbands must have same length, "
            f"got {len(lbands)} and {len(hbands)}"
        )
    
    # Validate frequency bands
    for i, (low, high) in enumerate(zip(lbands, hbands)):
        if low >= high:
            raise ValueError(
                f"Low frequency must be less than high frequency "
                f"for band {i}: {low} >= {high}"
            )
        if low < 0 or high < 0:
            raise ValueError(
                f"Frequencies must be non-negative for band {i}: "
                f"low={low}, high={high}"
            )


def synchronization(
    file_path: Union[str, Path],
    lbands: List[float],
    hbands: List[float],
    window_length: int,
    window_overlap: Union[int, float],
    sfreq: float = 256.0,
    phase_method: Union[str, PhaseMethod] = PhaseMethod.CSD,
    sync_method: Union[str, SyncMethod] = SyncMethod.WPLI,
    samples_per_epoch: Optional[int] = None,
    epochs_overlap: Optional[Union[int, float]] = None,
    use_numba: bool = True,
    uncache_data: bool = True,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
    apply_differential_window: bool = False,
    differential_window_param: float = 100000.0
) -> np.ndarray:
    """
    Compute synchronization measures from EEG data.
    
    This function loads EEG data, preprocesses it, and computes synchronization
    measures using either Cross-Spectral Density (CSD) or Hilbert transform
    methods for phase extraction.
    
    Args:
        file_path: Path to the EEG file (.edf or .bdf format)
        lbands: List of low frequency bounds for each band
        hbands: List of high frequency bounds for each band
        window_length: Length of processing windows in samples
        window_overlap: Overlap between windows (fraction < 1 or samples)
        sfreq: Sampling frequency in Hz (default: 256.0)
        phase_method: Phase extraction method ('csd' or 'hilbert')
        sync_method: Synchronization method (e.g., 'wpli', 'pli', 'plv')
        samples_per_epoch: Samples per epoch (required for CSD method)
        epochs_overlap: Overlap between epochs (required for CSD method)
        use_numba: Whether to use Numba acceleration (CSD method only)
        uncache_data: Whether to clear cached data after processing
        n_jobs: Number of parallel jobs (Hilbert method only)
        verbose: Whether to show progress information
        apply_differential_window: Whether to apply differential windowing
        differential_window_param: Parameter for differential windowing
        
    Returns:
        Synchronization matrix with shape (n_windows, n_bands, n_channels, n_channels)
        
    Raises:
        FileNotFoundError: If the EEG file doesn't exist
        ValueError: If parameters are invalid or incompatible
        ImportError: If required synchronization modules are not available
    """
    # Convert string parameters to enums
    if isinstance(phase_method, str):
        try:
            phase_method = PhaseMethod(phase_method.lower())
        except ValueError:
            raise ValueError(
                f"Unknown phase_method: {phase_method}. "
                f"Supported methods: {', '.join([m.value for m in PhaseMethod])}"
            )
    
    if isinstance(sync_method, str):
        sync_method = sync_method.lower()  # Keep as string for backward compatibility
    elif isinstance(sync_method, SyncMethod):
        sync_method = sync_method.value
    
    # Validate parameters
    _validate_synchronization_params(
        phase_method, samples_per_epoch, epochs_overlap,
        window_length, window_overlap, lbands, hbands
    )
    
    # Load and preprocess EEG data
    if verbose:
        print(f"Loading EEG data from: {file_path}")
    
    data, raw = _load_and_preprocess_eeg(file_path, verbose=False)
    
    if verbose:
        print(f"Loaded data shape: {data.shape}")
        print(f"Number of channels: {data.shape[1]}")
        print(f"Number of samples: {data.shape[0]}")
    
    # Apply differential windowing if requested
    if apply_differential_window:
        if verbose:
            print("Applying differential windowing...")
        # Transpose for differential_window function (expects channels, samples)
        data = data.T
        data = differential_window(data, differential_window_param)
        # Transpose back to (samples, channels)
        data = data.T
    
    # Compute synchronization based on phase method
    if verbose:
        print(f"Computing synchronization using {phase_method.value} method...")
    
    try:
        if phase_method == PhaseMethod.CSD:
            result = synchronization_csd.synchronization(
                data, lbands, hbands, window_length, window_overlap,
                samples_per_epoch, epochs_overlap,
                sfreq=sfreq,
                method=sync_method,
                uncache_data=uncache_data,
                use_numba=use_numba,
                verbose=verbose
            )
        
        elif phase_method == PhaseMethod.HILBERT:
            result = synchronization_hilbert.synchronization(
                data, lbands, hbands, window_length, window_overlap,
                n_jobs=n_jobs,
                sfreq=sfreq,
                method=sync_method,
                uncache_data=uncache_data,
                verbose=verbose
            )
        
        else:
            raise ValueError(f"Unsupported phase method: {phase_method}")
    
    except ImportError as e:
        raise ImportError(
            f"Required synchronization module not available: {e}. "
            f"Make sure synchronization_{phase_method.value} module is installed."
        )
    
    if verbose:
        print(f"Synchronization computation completed. Result shape: {result.shape}")
    
    return result


# Public API
__all__ = [
    "synchronization",
    "differential_window",
    "PhaseMethod",
    "SyncMethod",
]