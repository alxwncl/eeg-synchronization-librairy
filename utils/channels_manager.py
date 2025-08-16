import re
import numpy as np
from collections import Counter



def remove_duplicates(raw):
    base_names = [re.sub(r'-\d+$', '', ch) for ch in raw.ch_names]
    counts = Counter(base_names)

    for base, cnt in counts.items():
        if cnt > 1:
            # gather the actual channel names that share this base
            dup_chs = [ch for ch in raw.ch_names
                    if re.sub(r'-\d+$', '', ch) == base]
            # drop everything after the first
            to_drop = dup_chs[1:]
            raw.drop_channels(to_drop)
            to_rename = dup_chs[0]
            raw.rename_channels({to_rename: base})


def remove_bad_channels(raw):
    bad_re = re.compile(r"ECG|VNS|EKG1-CHIN|LOC-ROC|EKG1-EKG2|LUE-RAE")
    to_drop = [ch for ch in raw.ch_names if bad_re.search(ch)]
    raw.drop_channels(to_drop)


def remove_empty_channels(raw):
    to_drop = [ch for ch in raw.ch_names if ch.startswith('-') or ch == '']
    raw.drop_channels(to_drop)


def reorder_synchronization_matrix(sync_matrix, current_electrodes_order, reference_electrodes_order, verbose=False):
    """
    Optimized version: Reorders the last two dimensions of a matrix according to electrode ordering.
    Works with arrays of any dimensionality >= 2.
    
    Parameters:
    -----------
    sync_matrix : numpy.ndarray
        Input array with shape (..., n_electrodes, n_electrodes) where the last two 
        dimensions represent the synchronization matrix between electrodes
    current_electrodes_order : list
        Current ordering of electrodes corresponding to the last two dimensions
    reference_electrodes_order : list  
        Desired ordering of electrodes for the output
        
    Returns:
    --------
    numpy.ndarray
        Reordered array with shape (..., len(reference_electrodes_order), len(reference_electrodes_order))
    """
    # Convert to numpy array if not already
    sync_matrix = np.asarray(sync_matrix)
    
    # Validate inputs
    if sync_matrix.ndim < 2:
        raise ValueError(f"Input array must have at least 2 dimensions, got {sync_matrix.ndim}")
    
    if len(current_electrodes_order) != sync_matrix.shape[-1]:
        raise ValueError(f"Current electrode order length ({len(current_electrodes_order)}) "
                        f"doesn't match last matrix dimension ({sync_matrix.shape[-1]})")
   
    if sync_matrix.shape[-2] != sync_matrix.shape[-1]:
        raise ValueError(f"Last two dimensions must be equal (square matrix), "
                        f"got shape {sync_matrix.shape[-2:]} for last two dimensions")
    
    # Early return for identical orders
    if current_electrodes_order == reference_electrodes_order:
        return sync_matrix.copy()
   
    # Create mapping from electrode name to index (using dict comprehension)
    electrode_to_idx = {electrode: idx for idx, electrode in enumerate(current_electrodes_order)}
    
    # Convert to sets once for faster lookup
    current_set = set(current_electrodes_order)
    reference_set = set(reference_electrodes_order)
    
    # Find missing and extra electrodes
    missing_electrodes = reference_set - current_set
    extra_electrodes = current_set - reference_set
   
    if verbose:
        if missing_electrodes:
            print(f"Warning: Missing electrodes in current order will be filled with NaN: {missing_electrodes}")
        if extra_electrodes:
            print(f"Warning: Extra electrodes in current order will be ignored: {extra_electrodes}")
   
    # Vectorized index mapping - more efficient than list comprehension
    n_ref = len(reference_electrodes_order)
    reference_array = np.array(reference_electrodes_order)
    
    # Create boolean mask for valid electrodes
    valid_mask = np.array([electrode in electrode_to_idx for electrode in reference_electrodes_order])
    
    # Get indices for valid electrodes only
    if not np.any(valid_mask):
        # No valid electrodes - return all NaN
        output_shape = sync_matrix.shape[:-2] + (n_ref, n_ref)
        return np.full(output_shape, np.nan)
    
    # Map reference electrode names to current indices
    current_indices = np.array([electrode_to_idx[electrode] for electrode in reference_array[valid_mask]])
    valid_positions = np.where(valid_mask)[0]
    
    # Initialize output with NaN
    output_shape = sync_matrix.shape[:-2] + (n_ref, n_ref)
    reordered_matrix = np.full(output_shape, np.nan)
    
    # Optimized indexing: direct assignment without intermediate submatrix creation
    if len(current_indices) > 0:
        # Create meshgrid for 2D indexing
        curr_row_idx, curr_col_idx = np.meshgrid(current_indices, current_indices, indexing='ij')
        valid_row_idx, valid_col_idx = np.meshgrid(valid_positions, valid_positions, indexing='ij')
        
        # Build full indexing tuple for N-dimensional array
        leading_dims = tuple(slice(None) for _ in range(sync_matrix.ndim - 2))
        
        # Direct assignment - avoids creating intermediate arrays
        reordered_matrix[leading_dims + (valid_row_idx, valid_col_idx)] = sync_matrix[leading_dims + (curr_row_idx, curr_col_idx)]
   
    return reordered_matrix