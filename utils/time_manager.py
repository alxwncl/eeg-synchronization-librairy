from datetime import datetime, timedelta



def datetime_to_seconds(datetime_list):
    return [(dt - datetime_list[0]).total_seconds() for dt in datetime_list]


def parse_custom_time(time_str):
    """
    Parses a time string in '%H:%M:%S' format but allows for the hour
    to be 24 or higher. If the hour is >= 24, it subtracts 24 and returns a
    day_offset of 1 (indicating the time is on the next day), otherwise 0.
    
    Returns:
        (time_obj, day_offset): where time_obj is a datetime.time instance and
                                day_offset is an integer (0 or 1).
    """
    parts = time_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Time string '{time_str}' is not in the expected format HH:MM:SS")
    
    hour, minute, second = int(parts[0]), int(parts[1]), int(parts[2])
    day_offset = 0
    if hour >= 24:
        hour -= 24
        day_offset = 1  # This example assumes that the overflow will never be more than 24 hours.
    time_obj = datetime.strptime(f"{hour:02d}:{minute:02d}:{second:02d}", "%H:%M:%S").time()
    return time_obj, day_offset


def convert_seizures_times(seizure_sample_dict, record_time_dict, sample_rate):
    """
    Convert seizure sample indices to datetime objects using the record's start time.
    
    Parameters:
        seizure_sample_dict (dict): Dictionary mapping filenames to seizure events.
                                    Each event is a tuple (start_sample, end_sample)
                                    or None if there is no seizure.
        record_time_dict (dict): Dictionary mapping filenames to a tuple of 
                                 (record_start, record_end) as datetime objects.
        sample_rate (float or int): The number of samples per second.
        
    Returns:
        dict: A dictionary mapping each filename to a list of tuples containing 
              the seizure start and end times in datetime format. Files with no seizure 
              events will have a value of None.
    """
    seizure_datetime_dict = {}
    
    # Iterate over each record in the seizure dictionary.
    for record, seizures in seizure_sample_dict.items():
        # Get the start and end times for the record.
        time_tuple = record_time_dict.get(record)
        if time_tuple is None:
            # If the record time is missing, skip this record.
            continue
        
        record_start, _ = time_tuple

        # If there are no seizures, assign None.
        if seizures is None:
            seizure_datetime_dict[record] = None
        else:
            events = []
            # Process each seizure event.
            for start_sample, end_sample in seizures:
                # Convert sample indices to time offsets.
                start_time = record_start + timedelta(seconds=start_sample)
                end_time = record_start + timedelta(seconds=end_sample)
                events.append((start_time, end_time))
            seizure_datetime_dict[record] = events
            
    return seizure_datetime_dict


def compute_window_intervals(record_time_dict, window_size, stride=None):
    """
    Compute window boundaries for each record based on its start and end times.

    Parameters:
        record_time_dict (dict): Mapping from filename to a tuple (record_start, record_end),
                                 where both values are datetime objects.
        window_size (float or int): The duration of each window in seconds.
        window_stride (float or int, optional): The step (stride) between consecutive windows in seconds.
                                                Defaults to window_size (non-overlapping windows).
    
    Returns:
        dict: A dictionary where each key is a filename and each value is a list of tuples.
              Each tuple represents the (begin, end) datetime for a window.
              
    Notes:
        - Only full windows (where window_end <= record_end) are included.
        - To have overlapping windows, choose a window_stride smaller than window_size.
        - If no full window fits in a record, the value for that record will be an empty list.
    """
    # Use non-overlapping windows if no stride is explicitly provided.
    if stride is None:
        stride = window_size

    # Create timedelta objects for window size and stride.
    window_delta = timedelta(seconds=window_size)
    stride_delta = timedelta(seconds=stride)
    
    window_dict = {}
    
    # For each record, compute the windows.
    for filename, time_tuple in record_time_dict.items():
        record_start, record_end = time_tuple
        windows = []
        current_start = record_start
        
        # While the window (current_start + window_size) is within the record duration.
        while current_start + window_delta <= record_end:
            window_begin = current_start
            window_end = current_start + window_delta
            windows.append((window_begin, window_end))
            current_start += stride_delta
        
        window_dict[filename] = windows[:-1]
        
    return window_dict