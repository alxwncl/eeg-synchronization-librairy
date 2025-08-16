import re

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mne")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

from . import parse_chb, parse_private
from .channels_manager import remove_duplicates, remove_bad_channels, remove_empty_channels, reorder_synchronization_matrix
from .time_manager import datetime_to_seconds, convert_seizures_times, compute_window_intervals
from .data_manager import SqliteDict


def find_seizures_times(dir_path):
    dataset_pattern = re.compile(r"chb")
    if dataset_pattern.search(dir_path):
        return parse_chb.find_seizures_times(dir_path)
    else:
        return parse_private.find_seizures_times(dir_path)
    
def find_records_times(dir_path):
    dataset_pattern = re.compile(r"chb")
    if dataset_pattern.search(dir_path):
        return parse_chb.find_records_times(dir_path)
    else:
        return parse_private.find_records_times(dir_path)
    
def find_electrodes_placements(dir_path, normalize_channels=True):
    dataset_pattern = re.compile(r"chb")
    if dataset_pattern.search(dir_path):
        return parse_chb.find_electrodes_placements(dir_path, normalize_channels=normalize_channels)
    else:
        return parse_private.find_electrodes_placements(dir_path, normalize_channels=normalize_channels)

    

    
def process_eeg_files(dir_path, process_func, cache_name=None, header=None, verbose=True, max_workers=None):
    dataset_pattern = re.compile(r"chb")
    if dataset_pattern.search(dir_path):
        return parse_chb.process_eeg_files(dir_path, process_func, cache_name=cache_name, header=header, verbose=verbose, max_workers=max_workers)
    else:
        return parse_private.process_eeg_files(dir_path, process_func, cache_name=cache_name, header=header, verbose=verbose, max_workers=max_workers)

def create_header(phase_method, sync_method, window_length, window_overlap, lbands, hbands, epochs_per_window=None, samples_per_epoch=None, epochs_overlap=None):
    if phase_method == "csd":
        if samples_per_epoch is None or epochs_overlap is None or epochs_per_window is None:
            raise ValueError("samples_per_epoch, epochs_per_window and epochs_overlap must be provided for csd phase method.")
        return (
                f"Phase method: {phase_method}\n"
                f"Synchronization method: {sync_method}\n"
                f"Samples per window: {window_length}\n"
                f"Samples per epoch: {samples_per_epoch}\n"
                f"Epochs overlap: {epochs_overlap}\n"
                f"Epochs per window: {epochs_per_window}\n"
                f"Shared epochs: {window_overlap}\n"
                f"Low bands: {lbands}\n"
                f"High bands: {hbands}\n"
            )
    elif phase_method == "hilbert":
        return (
                f"Phase method: {phase_method}\n"
                f"Synchronization method: {sync_method}\n"
                f"Samples per window: {window_length}\n"
                f"Windows_overlap: {window_overlap}\n"
                f"Low bands: {lbands}\n"
                f"High bands: {hbands}\n"
            )
    else:
        raise ValueError(f"Unknown phase method: {phase_method}")
    
def create_cache_name(phase_method, sync_method, window_length, overlap, epochs_per_window=None):
    overlap_percent = int(overlap)
    if overlap <= 1:
        overlap_percent = int(overlap * 100)
    if phase_method == "csd":
        if epochs_per_window is None:
            raise ValueError("epochs_per_window must be provided for csd phase method.")
        return f"{phase_method}_{sync_method}_{window_length}_{epochs_per_window}_{overlap_percent}"
    elif phase_method == "hilbert":
        return f"{phase_method}_{sync_method}_{window_length}_{overlap_percent}"
    else:
        raise ValueError(f"Unknown phase method: {phase_method}")
    
def parse_header(header):
    """
    Parse the header string to extract parameters.
    """
    params = {}
    for line in header.split("\n"):
        if line.strip():
            key, value = line.split(": ")
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.startswith("[") and value.endswith("]"):
                        value = [float(x) for x in value[1:-1].split(",")]
                    else:
                        value = str(value)
            params[key] = value
    return params
    

__all__ = ["find_seizures_times", "find_records_times", "find_electrodes_placements", "process_eeg_files", "datetime_to_seconds", 
           "convert_seizures_times", "compute_window_intervals", "remove_duplicates", "remove_bad_channels", "remove_empty_channels", 
           "reorder_synchronization_matrix", "create_header", "create_cache_name", "parse_header", "SqliteDict"]
