import os
import re
import mne
import time
import psutil
import pickle
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from .channels_manager import remove_duplicates, remove_bad_channels, remove_empty_channels
from .time_manager import parse_custom_time
from .data_manager import SqliteDict
from .ram_manager import RAMMonitor



def find_seizures_times(dir_path):
    """
    Scans `dir_path` for:
      - EDF files named like '190219a-c_0000.edf', '190219a-c_0001.edf', …
      - Annotation files named like '190219a-c.txt'
    
    Returns a dict mapping each EDF filename → list of seizure onset times
    (in seconds since the start of that EDF), or None if no seizure in that file.
    """
    # Collect and sort EDF filenames
    edf_files = sorted(
        fname
        for fname in os.listdir(dir_path)
        if fname.lower().endswith('.edf')
    )

    # Initialize output
    all_seizures = {edf.lower(): None for edf in edf_files}

    # Pattern of time in annotations
    time_pattern = re.compile(r'(\d{1,2})h(\d{1,2})m(\d{1,2})s')

    # Process annotations
    for txt in os.listdir(dir_path):
        if not txt.lower().endswith('.txt'):
            continue
        base = os.path.splitext(txt)[0].lower()
        path = os.path.join(dir_path, txt).lower()

        # Read all elapsed times
        sessions_seizures = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                times = time_pattern.search(line)
                if not times:
                    continue
                h, m, s = map(int, times.groups())
                # None for no seizure offset
                sessions_seizures.append((h*3600 + m*60 + s, None))

        # Skip if no seizure annotations
        if not sessions_seizures:
            continue

        # Bucket each seizure into its segment
        for t in sessions_seizures:
            seg_idx = t // 3600
            offset = t - seg_idx*3600
            edf_name = f"{base}_{seg_idx:04d}.edf"
            if edf_name in all_seizures:
                if all_seizures[edf_name] is None:
                    all_seizures[edf_name] = []
                all_seizures[edf_name].append(offset)
            else:
                print(f"Warning: {edf_name} not found on disk.")
                pass

    return all_seizures


def find_records_times(dir_path):
    """
    Scans `dir_path` for:
      - EDF files named like '190219a-c_0000.edf', '190219a-c_0001.edf', …
    
    Returns a dict mapping each EDF filename → (segment_start, segment_end)
    datetime tuples. The start of the first segment of each session is set
    to midnight of the session date as a dummy timestamp.
    """
    # Collect all EDF filenames
    edf_files = sorted(
        fname
        for fname in os.listdir(dir_path)
        if fname.lower().endswith('.edf')
    )

    # Initialize output for each EDF
    record_times = {}

    # Group EDFs by session base name
    sessions = {}
    for fname in edf_files:
        base, idx = os.path.splitext(fname)[0].rsplit('_', 1)
        sessions.setdefault(base, []).append((int(idx), fname))

    # Process each session
    for base, segments in sessions.items():
        # sort segments by index
        segments.sort(key=lambda x: x[0])

        # parse session date from base (YYMMDD)
        yy = int(base[:2])
        mm = int(base[2:4])
        dd = int(base[4:6])
        year = 2000 + yy if yy < 70 else 1900 + yy  # adjust century if needed
        # dummy start: midnight of session date
        session_start = datetime(year, mm, dd)

        # assign times for each segment
        for idx, fname in segments:
            segment_start = session_start + timedelta(hours=idx)
            segment_end   = segment_start + timedelta(hours=1)
            record_times[fname] = (segment_start, segment_end)

    return record_times



def find_electrodes_placements(dir_path, normalize_channels=True):
    """
    Find electrode placements for .edf files in a private dataset directory.
    
    Parameters:
        dir_path (str or Path): Path to the directory containing .edf files.
                               Expected to follow format with "Pat" in the name.
        normalize_channels (bool): Whether to apply channel normalization 
                                  (remove duplicates, bad channels, empty channels).
                                  Default is True.
    
    Returns:
        dict: Dictionary where keys are filenames and values are lists of channel names.
              Format: {filename.edf: [channel1, channel2, ...]}
    
    Raises:
        ValueError: If the directory name doesn't contain "Pat" identifier.
        FileNotFoundError: If the directory doesn't exist.
    """
    dir = Path(dir_path)
    
    # Validate directory exists
    if not dir.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Extract patient ID from directory path
    dir_name = dir.name
    if "Pat" not in dir_name:
        raise ValueError(f"Invalid directory name: {dir_name}. Expected directory name to contain 'Pat'")
    
    patient_id = dir_name.split("Pat")[-1]
    
    output = {}
    
    # Process all .edf files in the directory
    for file in dir.iterdir():
        if file.is_file() and file.suffix.lower() == '.edf':
            try:
                # Read EDF file header only (preload=False for efficiency)
                raw = mne.io.read_raw_edf(file, preload=False, stim_channel=None, verbose=False)
                
                # Apply channel normalization if requested
                if normalize_channels:
                    remove_duplicates(raw)
                    remove_bad_channels(raw)
                    remove_empty_channels(raw)
                
                # Store channel names for this file
                output[file.name] = raw.info["ch_names"]
                
            except Exception as e:
                print(f"Warning: Could not process {file.name}: {e}")
                # Continue processing other files even if one fails
                continue
    
    return output


def process_eeg_files(dir_path, process_func, cache_name=None, header=None, verbose=True, max_workers=None):
    """
    Processes only the .edf (or .EDF) files in dir_path in parallel.
    Any other extensions (e.g. .eeg) are skipped.
    Results are cached to avoid re-processing.
    Parameters:
        dir_path (str): directory with mixed .edf/.eeg files
        process_func (callable): fn(path_to_edf) → result
        cache_name (str, optional): cache file suffix
        header (any, optional): if you pass a header, it must match the one in cache
        verbose (bool): show progress/logging
        max_workers (int, optional): maximum number of worker threads
    Returns:
        dict: { filename.edf → process_func_result }
    """
    patient_id = dir_path.split("Pat")[-1]
    if cache_name is None:
        cache_file = os.path.join(dir_path, f"processedeeg{patient_id}.pkl")
    else:
        cache_file = os.path.join(dir_path, f"processedeeg{patient_id}{cache_name}.pkl")
    
    # Initialize cache_data
    cache_data = {"header": header}
    files_to_process = []
    
    # Load existing cache if it exists and header matches
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            if cached_data.get("header", None) == header:
                cache_data = cached_data
                if verbose:
                    print("Loaded cached processed files.")
            else:
                if verbose:
                    print("Cache header mismatch. Recreating cache.")
        except Exception as e:
            if verbose:
                print(f"Failed to load cache file: {e}. Recreating cache.")
    
    # Get sorted list of EDF files
    edf_files = sorted(
        fn for fn in os.listdir(dir_path)
        if fn.lower().endswith(".edf")
    )
    
    # Determine which files need processing (not in cache or cache is empty/invalid)
    for fn in edf_files:
        if fn not in cache_data or cache_data.get("header", None) != header:
            files_to_process.append(fn)
    
    if not files_to_process:
        if verbose:
            print("All files already processed and cached.")
        return cache_data
    
    # Process files in parallel
    def process_single_file(filename):
        """Helper function to process a single file"""
        path = os.path.join(dir_path, filename)
        try:
            result = process_func(path)
            return filename, result, None  # filename, result, error
        except Exception as e:
            return filename, None, str(e)  # filename, result, error
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all files for processing
        future_to_filename = {
            executor.submit(process_single_file, fn): fn 
            for fn in files_to_process
        }
        
        # Process completed futures with progress bar
        iterator = tqdm(
            as_completed(future_to_filename), 
            total=len(files_to_process),
            desc="Processing files", 
            disable=not verbose
        )
        
        for future in iterator:
            filename = future_to_filename[future]
            try:
                fn, result, error = future.result()
                if error is None:
                    cache_data[fn] = result
                    if verbose:
                        iterator.set_postfix_str(f"✓ {fn}")
                else:
                    if verbose:
                        print(f"Error processing {fn}: {error}")
                        iterator.set_postfix_str(f"✗ {fn}")
            except Exception as e:
                if verbose:
                    print(f"Unexpected error processing {filename}: {e}")
    
    # Write updated cache
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        if verbose:
            print(f"Cache saved with {len(files_to_process)} newly processed files.")
    except Exception as e:
        if verbose:
            print(f"Failed to save cache file: {e}")
    
    return cache_data