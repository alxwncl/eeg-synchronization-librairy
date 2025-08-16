"""
EEG Data Processing Module

This module provides utilities for processing EEG datasets, including
seizure detection, electrode placement management, and data processing
with various phase and synchronization methods.
"""

import re
import warnings
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Callable, Tuple
from enum import Enum

# Suppress specific MNE warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mne")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

from . import parse_chb, parse_private
from .channels_manager import (
    remove_duplicates, 
    remove_bad_channels, 
    remove_empty_channels, 
    reorder_synchronization_matrix
)
from .time_manager import (
    datetime_to_seconds, 
    convert_seizures_times, 
    compute_window_intervals
)
from .data_manager import SqliteDict


class PhaseMethod(Enum):
    """Supported phase computation methods."""
    CSD = "csd"
    HILBERT = "hilbert"


class DatasetType(Enum):
    """Supported dataset types."""
    CHB = "chb"
    PRIVATE = "private"


def _detect_dataset_type(dir_path: Union[str, Path]) -> DatasetType:
    """
    Detect the dataset type based on directory path.
    
    Args:
        dir_path: Path to the dataset directory
        
    Returns:
        DatasetType enum value
    """
    path_str = str(dir_path)
    if re.search(r"chb", path_str, re.IGNORECASE):
        return DatasetType.CHB
    return DatasetType.PRIVATE


def find_seizures_times(dir_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Find seizure times in the dataset.
    
    Args:
        dir_path: Path to the dataset directory
        
    Returns:
        Dictionary containing seizure timing information
    """
    dataset_type = _detect_dataset_type(dir_path)
    
    if dataset_type == DatasetType.CHB:
        return parse_chb.find_seizures_times(dir_path)
    else:
        return parse_private.find_seizures_times(dir_path)


def find_records_times(dir_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Find recording times in the dataset.
    
    Args:
        dir_path: Path to the dataset directory
        
    Returns:
        Dictionary containing recording timing information
    """
    dataset_type = _detect_dataset_type(dir_path)
    
    if dataset_type == DatasetType.CHB:
        return parse_chb.find_records_times(dir_path)
    else:
        return parse_private.find_records_times(dir_path)


def find_electrodes_placements(
    dir_path: Union[str, Path], 
    normalize_channels: bool = True
) -> Dict[str, Any]:
    """
    Find electrode placements in the dataset.
    
    Args:
        dir_path: Path to the dataset directory
        normalize_channels: Whether to normalize channel names
        
    Returns:
        Dictionary containing electrode placement information
    """
    dataset_type = _detect_dataset_type(dir_path)
    
    if dataset_type == DatasetType.CHB:
        return parse_chb.find_electrodes_placements(
            dir_path, 
            normalize_channels=normalize_channels
        )
    else:
        return parse_private.find_electrodes_placements(
            dir_path, 
            normalize_channels=normalize_channels
        )


def process_eeg_files(
    dir_path: Union[str, Path],
    process_func: Callable,
    cache_name: Optional[str] = None,
    header: Optional[str] = None,
    verbose: bool = True,
    max_workers: Optional[int] = None
) -> Any:
    """
    Process EEG files in the dataset.
    
    Args:
        dir_path: Path to the dataset directory
        process_func: Function to apply to each EEG file
        cache_name: Name for caching results
        header: Header information for processing
        verbose: Whether to show progress information
        max_workers: Maximum number of worker threads
        
    Returns:
        Processing results
    """
    dataset_type = _detect_dataset_type(dir_path)
    
    if dataset_type == DatasetType.CHB:
        return parse_chb.process_eeg_files(
            dir_path, 
            process_func, 
            cache_name=cache_name,
            header=header,
            verbose=verbose,
            max_workers=max_workers
        )
    else:
        return parse_private.process_eeg_files(
            dir_path,
            process_func,
            cache_name=cache_name,
            header=header,
            verbose=verbose,
            max_workers=max_workers
        )


def create_header(
    phase_method: Union[str, PhaseMethod],
    sync_method: str,
    window_length: int,
    window_overlap: Union[int, float],
    lbands: List[float],
    hbands: List[float],
    epochs_per_window: Optional[int] = None,
    samples_per_epoch: Optional[int] = None,
    epochs_overlap: Optional[Union[int, float]] = None
) -> str:
    """
    Create a header string with processing parameters.
    
    Args:
        phase_method: Phase computation method ('csd' or 'hilbert')
        sync_method: Synchronization method
        window_length: Length of processing windows
        window_overlap: Overlap between windows
        lbands: Low frequency bands
        hbands: High frequency bands
        epochs_per_window: Number of epochs per window (required for CSD)
        samples_per_epoch: Samples per epoch (required for CSD)
        epochs_overlap: Overlap between epochs (required for CSD)
        
    Returns:
        Formatted header string
        
    Raises:
        ValueError: If required parameters are missing for the specified method
    """
    # Convert string to enum if necessary
    if isinstance(phase_method, str):
        try:
            phase_method = PhaseMethod(phase_method.lower())
        except ValueError:
            raise ValueError(f"Unknown phase method: {phase_method}")
    
    if phase_method == PhaseMethod.CSD:
        # Validate required parameters for CSD
        missing_params = []
        if samples_per_epoch is None:
            missing_params.append("samples_per_epoch")
        if epochs_overlap is None:
            missing_params.append("epochs_overlap")
        if epochs_per_window is None:
            missing_params.append("epochs_per_window")
            
        if missing_params:
            raise ValueError(
                f"The following parameters are required for CSD phase method: "
                f"{', '.join(missing_params)}"
            )
        
        return (
            f"Phase method: {phase_method.value}\n"
            f"Synchronization method: {sync_method}\n"
            f"Samples per window: {window_length}\n"
            f"Samples per epoch: {samples_per_epoch}\n"
            f"Epochs overlap: {epochs_overlap}\n"
            f"Epochs per window: {epochs_per_window}\n"
            f"Shared epochs: {window_overlap}\n"
            f"Low bands: {lbands}\n"
            f"High bands: {hbands}\n"
        )
    
    elif phase_method == PhaseMethod.HILBERT:
        return (
            f"Phase method: {phase_method.value}\n"
            f"Synchronization method: {sync_method}\n"
            f"Samples per window: {window_length}\n"
            f"Windows_overlap: {window_overlap}\n"
            f"Low bands: {lbands}\n"
            f"High bands: {hbands}\n"
        )
    
    else:
        raise ValueError(f"Unknown phase method: {phase_method}")


def create_cache_name(
    phase_method: Union[str, PhaseMethod],
    sync_method: str,
    window_length: int,
    overlap: Union[int, float],
    epochs_per_window: Optional[int] = None
) -> str:
    """
    Create a cache name based on processing parameters.
    
    Args:
        phase_method: Phase computation method
        sync_method: Synchronization method
        window_length: Length of processing windows
        overlap: Overlap between windows (as percentage or fraction)
        epochs_per_window: Number of epochs per window (required for CSD)
        
    Returns:
        Cache name string
        
    Raises:
        ValueError: If required parameters are missing for the specified method
    """
    # Convert string to enum if necessary
    if isinstance(phase_method, str):
        try:
            phase_method = PhaseMethod(phase_method.lower())
        except ValueError:
            raise ValueError(f"Unknown phase method: {phase_method}")
    
    # Convert overlap to percentage if it's a fraction
    overlap_percent = int(overlap) if overlap > 1 else int(overlap * 100)
    
    if phase_method == PhaseMethod.CSD:
        if epochs_per_window is None:
            raise ValueError(
                "epochs_per_window must be provided for CSD phase method"
            )
        return (
            f"{phase_method.value}_{sync_method}_{window_length}_"
            f"{epochs_per_window}_{overlap_percent}"
        )
    
    elif phase_method == PhaseMethod.HILBERT:
        return (
            f"{phase_method.value}_{sync_method}_{window_length}_{overlap_percent}"
        )
    
    else:
        raise ValueError(f"Unknown phase method: {phase_method}")


def parse_header(header: str) -> Dict[str, Any]:
    """
    Parse a header string to extract parameters.
    
    Args:
        header: Header string containing parameter information
        
    Returns:
        Dictionary of parsed parameters
        
    Raises:
        ValueError: If header format is invalid
    """
    params = {}
    
    for line_num, line in enumerate(header.split("\n"), 1):
        line = line.strip()
        if not line:
            continue
            
        if ": " not in line:
            raise ValueError(
                f"Invalid header format at line {line_num}. "
                f"Expected 'key: value', got: '{line}'"
            )
        
        key, value_str = line.split(": ", 1)
        key = key.strip()
        value_str = value_str.strip()
        
        # Parse the value
        try:
            params[key] = _parse_value(value_str)
        except ValueError as e:
            raise ValueError(
                f"Error parsing value '{value_str}' for key '{key}' "
                f"at line {line_num}: {e}"
            )
    
    return params


def _parse_value(value_str: str) -> Union[int, float, List[float], str]:
    """
    Parse a string value to appropriate Python type.
    
    Args:
        value_str: String representation of the value
        
    Returns:
        Parsed value (int, float, list, or str)
    """
    # Try to parse as integer
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # Try to parse as float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Try to parse as list
    if value_str.startswith("[") and value_str.endswith("]"):
        try:
            # Remove brackets and split by comma
            list_content = value_str[1:-1].strip()
            if not list_content:  # Empty list
                return []
            return [float(x.strip()) for x in list_content.split(",")]
        except ValueError:
            pass
    
    # Return as string if all else fails
    return value_str


# Public API
__all__ = [
    # Core functions
    "find_seizures_times",
    "find_records_times", 
    "find_electrodes_placements",
    "process_eeg_files",
    
    # Header and cache utilities
    "create_header",
    "create_cache_name", 
    "parse_header",
    
    # Time management
    "datetime_to_seconds",
    "convert_seizures_times", 
    "compute_window_intervals",
    
    # Channel management
    "remove_duplicates",
    "remove_bad_channels",
    "remove_empty_channels", 
    "reorder_synchronization_matrix",
    
    # Data management
    "SqliteDict",
    
    # Enums
    "PhaseMethod",
    "DatasetType",
]