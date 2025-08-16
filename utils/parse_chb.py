import re
import os
import mne
import time
import psutil
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from .channels_manager import remove_duplicates, remove_bad_channels, remove_empty_channels
from .time_manager import parse_custom_time
from .data_manager import SqliteDict
from .ram_manager import RAMMonitor


def find_seizures_times(dir_path):

    # Extract events file
    dir_pattern = re.compile(r'^chb\d{2}$')
    dir_name = Path(dir_path).name
    if not dir_pattern.match(dir_name):
        raise ValueError(f"Invalid dir_path name: {dir_name}. Expected format: chb_XX")
    else:
        patient_id = dir_name.split('b')[1]
        file_path = dir_path + f'\\chb{patient_id}-summary.txt'

    # Dictionary to store the results
    results = {}
    
    # Regex patterns to match both "Seizure Start Time:" and "Seizure 1 Start Time:" formats
    seizure_start_pattern = re.compile(r"Seizure(?: \d+)? Start Time:\s*(\d+)")
    seizure_end_pattern   = re.compile(r"Seizure(?: \d+)? End Time:\s*(\d+)")
    
    with open(file_path, 'r') as file:
        # Create an iterator over the lines of the file
        lines = iter(file)
        for line in lines:
            line = line.strip()
            # Only process lines we care about
            if line.startswith("File Name:"):
                current_file = line.split(":", 1)[1].strip()
                # Default entry: if there are no seizures, we'll leave it as None
                results[current_file] = None

            elif line.startswith("Number of Seizures in File:"):
                count_str = line.split(":", 1)[1].strip()
                try:
                    seizure_count = int(count_str)
                except ValueError:
                    seizure_count = 0

                # If there are seizures, read the expected number of seizure time pairs
                if seizure_count > 0:
                    seizures = []
                    for _ in range(seizure_count):
                        # Read the seizure start time line
                        start_line = next(lines).strip()
                        match_start = seizure_start_pattern.search(start_line)
                        if match_start:
                            seizure_start = int(match_start.group(1))
                        else:
                            raise ValueError(f"Expected a seizure start time line, got: {start_line}")

                        # Read the seizure end time line
                        end_line = next(lines).strip()
                        match_end = seizure_end_pattern.search(end_line)
                        if match_end:
                            seizure_end = int(match_end.group(1))
                        else:
                            raise ValueError(f"Expected a seizure end time line, got: {end_line}")
                        
                        seizures.append((seizure_start, seizure_end))
                    results[current_file] = seizures
                    
    return results


def find_records_times(dir_path):
    """
    Reads a summary file and extracts each record's start and end times, converting them into datetime
    objects while preserving the order even if records cross midnight. This function handles time strings
    that may contain hours >= 24 (e.g., "24:44:29") by interpreting them as next-day times.
    
    Expected record format in the file:
        File Name: <filename>
        File Start Time: <start time>   # e.g., "23:50:00" or "24:44:29"
        File End Time: <end time>       # e.g., "00:15:00"
    
    Returns:
        A dictionary mapping each file name (str) to a tuple:
            (start_datetime, end_datetime)
        where start_datetime and end_datetime are complete datetime objects that maintain the proper
        sequential order.
    """

    # Extract events file
    dir_pattern = re.compile(r'^chb\d{2}$')
    dir_name = Path(dir_path).name
    if not dir_pattern.match(dir_name):
        raise ValueError(f"Invalid dir_path name: {dir_name}. Expected format: chb_XX")
    else:
        patient_id = dir_name.split('b')[1]
        file_path = dir_path + f'\\chb{patient_id}-summary.txt'

    # Dictionary to store the final parsed results.
    results = {}
    current_file = None
    file_start_str = None

    # Set an arbitrary base date for the first record.
    base_date = datetime(2000, 1, 1)
    current_date = base_date
    previous_end_dt = None  # This will track the end time of the previous record

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Identify the record's filename.
            if line.startswith("File Name:"):
                current_file = line.split(":", 1)[1].strip()
            # Capture the record's start time string.
            elif line.startswith("File Start Time:"):
                file_start_str = line.split(":", 1)[1].strip()
            # When encountering the end time, process the record.
            elif line.startswith("File End Time:"):
                file_end_str = line.split(":", 1)[1].strip()
                if current_file is not None and file_start_str is not None:
                    # Convert the start and end time strings using the helper function.
                    start_time, start_offset = parse_custom_time(file_start_str)
                    end_time, end_offset = parse_custom_time(file_end_str)
                    
                    # Create datetime objects by combining the current base date and applying offsets.
                    start_dt = datetime.combine(current_date, start_time) + timedelta(days=start_offset)
                    end_dt = datetime.combine(current_date, end_time) + timedelta(days=end_offset)
                    
                    # If the new record's start is earlier than the previous record's end,
                    # assume that the record has rolled into the next day.
                    if previous_end_dt is not None and start_dt < previous_end_dt:
                        current_date += timedelta(days=1)
                        start_dt = datetime.combine(current_date, start_time) + timedelta(days=start_offset)
                        end_dt = datetime.combine(current_date, end_time) + timedelta(days=end_offset)
                    
                    results[current_file] = (start_dt, end_dt)
                    previous_end_dt = end_dt  # Update for next iteration
                    
                    # Reset temporary variables for the next record.
                    current_file = None
                    file_start_str = None

    return results


def find_electrodes_placements(dir_path, normalize_channels=True):
    # Extract .edf files
    dir_pattern = re.compile(r'^chb\d{2}$')
    dir = Path(dir_path)
    dir_name = dir.name
    if not dir_pattern.match(dir_name):
        raise ValueError(f"Invalid dir_path name: {dir_name}. Expected format: chb_XX")
    else:
        patient_id = dir_name.split('b')[1]

    file_pattern = re.compile(rf"^chb{patient_id}_(\d+)\.edf$", re.IGNORECASE)

    
    output = {}
    for file in dir.iterdir():
        if file.is_file():
            if file_pattern.match(file.name):
                raw = mne.io.read_raw_edf(file, preload=False, stim_channel=None, verbose=False)
                if normalize_channels:
                    remove_duplicates(raw)
                    remove_bad_channels(raw)
                    remove_empty_channels(raw)
                output[file.name] = raw.info["ch_names"]
    
    return output

                


def process_eeg_files(dir_path, process_func, cache_name=None, header=None, verbose=True, 
                     max_workers=None, max_memory_percent=85, memory_check_interval=0.5):
    """
    Processes EEG record files in a specified dir_path for a given patient number with RAM monitoring.

    Files must follow the naming format: "chbXX_YY.edf", where:
      - XX: Patient number (formatted as two digits, e.g., "04", "32")
      - YY: Record number (one or more digits)

    Other files in the dir_path are ignored.

    Additionally, this function caches processed results in a SQLite database in the dir_path.
    If the cache file exists and its header matches the provided header, the cached results are loaded.
    If the header differs or if the cache file cannot be loaded, the cache is recreated.

    Parameters:
        dir_path (str): Path to the dir_path containing the EEG files.
        process_func (function): A function to apply to each valid EEG file.
                                 This function should take one argument: the file's full path.
        cache_name (str, optional): Name suffix for the cache file. Default is None.
        header (any, optional): A header to compare with the cache file's header. Default is None.
        verbose (bool): If True, display progress and status messages. Default is True.
        max_workers (int, optional): Maximum number of parallel workers. Default is None (uses CPU count - 1).
        max_memory_percent (float): Maximum RAM usage percentage before blocking new tasks. Default is 85.
        memory_check_interval (float): How often to check RAM usage in seconds. Default is 0.5.

    Returns:
        dict: A dictionary of the results returned by process_func for each processed file.
              The keys are the filenames and the values are the results from process_func.
    """
    # Ensure the patient number is a two-digit string
    patient_id = dir_path.split('chb')[-1]

    # Create a regular expression pattern for the filenames
    pattern = re.compile(rf"^chb{patient_id}(\d+)\.edf$")

    # Define a cache filename for previously processed files
    if cache_name is None:
        cache_file = os.path.join(dir_path, f"processedeeg{patient_id}.db")
    else:
        cache_file = os.path.join(dir_path, f"processedeeg{patient_id}{cache_name}.db")

    # Initialize RAM monitor
    ram_monitor = RAMMonitor(max_memory_percent, memory_check_interval, verbose)

    # Initialize SQLite cache
    with SqliteDict(cache_file) as cache:
        # Check cache header
        cached_header = cache.get("header", None)
        if cached_header == header and cached_header is not None:
            if verbose:
                print("Loaded cached processed files.")
            # Return all cached results as a regular dictionary
            return dict(cache.items())
        else:
            if verbose and os.path.exists(cache_file):
                print("Cache header mismatch. Recreating cache.")
            # Clear cache and set new header
            cache.conn.execute('DELETE FROM cache')
            cache.conn.commit()
            cache["header"] = header

        # Get list of files to process
        files_to_process = []
        for filename in os.listdir(dir_path):
            if pattern.match(filename) and filename not in cache:
                filepath = os.path.join(dir_path, filename)
                files_to_process.append((filename, filepath))

        if not files_to_process:
            if verbose:
                print("No files to process.")
            return dict(cache.items())

        # Set max_workers based on CPU count and RAM considerations
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)

        # Adjust max_workers based on available RAM
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        if available_ram_gb < 4:  # Less than 4GB available
            max_workers = min(max_workers, 2)
        elif available_ram_gb < 8:  # Less than 8GB available
            max_workers = min(max_workers, max_workers // 2)

        if verbose:
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            print(f"RAM-aware processing: {max_workers} workers, {available_ram_gb:.1f}GB/{total_ram_gb:.1f}GB available")

        # Process files in parallel with ThreadPoolExecutor and RAM monitoring
        def process_single_file(filename_filepath):
            filename, filepath = filename_filepath
            try:
                result = process_func(filepath)
                return filename, result, None
            except Exception as e:
                return filename, None, e

        with ThreadPoolExecutor(max_workers=4) as executor:
            submitted_futures = {}
            completed_count = 0
            files_iter = iter(files_to_process)

            # Initialize progress bar
            if verbose:
                pbar = tqdm(total=len(files_to_process), desc="Processing files")

            while completed_count < len(files_to_process):
                # Submit new tasks if RAM allows and we have work to do
                while (len(submitted_futures) < max_workers and 
                       ram_monitor.can_submit_task()):
                    try:
                        file_info = next(files_iter)
                        future = executor.submit(process_single_file, file_info)
                        submitted_futures[future] = file_info[0]
                    except StopIteration:
                        break  # No more files to process

                # If we can't submit due to RAM, wait a bit
                if not ram_monitor.can_submit_task() and len(submitted_futures) == max_workers:
                    if verbose and ram_monitor.get_memory_usage() > max_memory_percent * 0.9:
                        memory_percent = ram_monitor.get_memory_usage()
                        # print(f"\nRAM usage high ({memory_percent:.1f}%), waiting for tasks to complete...")

                # Process completed tasks
                if submitted_futures:
                    # Wait for at least one task to complete
                    done_futures = []
                    try:
                        for future in as_completed(submitted_futures, timeout=1.0):
                            done_futures.append(future)
                            break  # Process one at a time to check RAM more frequently
                    except TimeoutError:
                        # No futures completed within timeout, continue to next iteration
                        pass

                    for future in done_futures:
                        filename = submitted_futures.pop(future)
                        filename_result, result, error = future.result()

                        if error:
                            print(f"Error processing {filename}: {error}")
                        else:
                            # Save result immediately to cache (incremental update)
                            cache[filename_result] = result

                        completed_count += 1
                        if verbose:
                            pbar.update(1)

                # Small delay to prevent busy waiting
                if not submitted_futures:
                    time.sleep(0.1)

            if verbose:
                pbar.close()

        # Return all results as a regular dictionary
        return dict(cache.items())