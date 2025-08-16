# EEG Synchronization Analysis

A Python library for computing synchronization measures from EEG data using different phase extraction methods and various connectivity metrics.

## Features

- **Multiple Phase Extraction Methods**: Cross-Spectral Density (CSD) and Hilbert transform
- **Synchronization Metrics**: WPLI, PLI, PLV, dPLI, and Coherence
- **Dataset Support**: CHB-MIT and private EEG datasets
- **Parallel Processing**: Multi-threaded computation with memory management
- **Caching**: SQLite-based result caching for efficient reprocessing
- **Channel Management**: Automatic handling of duplicate, bad, and empty channels

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- `mne==1.10.1` - EEG data processing
- `numpy==2.3.2` - Numerical computations
- `scipy==1.16.1` - Signal processing
- `numba==0.61.2` - Performance optimization
- `joblib==1.4.2` - Parallel processing
- `tqdm==4.67.1` - Progress bars
- `psutil==7.0.0` - System monitoring

## Quick Start

### Basic Synchronization Analysis

```python
from synchronization import synchronization, PhaseMethod, SyncMethod

# Define parameters
file_path = 'path/to/your/eeg_file.edf'
lbands = [8, 13, 30]      # Low frequency bounds
hbands = [12, 30, 100]    # High frequency bounds
window_length = 1280      # Window length in samples
window_overlap = 0.5      # 50% overlap between windows

# For Hilbert method
sync_matrix = synchronization(
    file_path=file_path,
    lbands=lbands,
    hbands=hbands,
    window_length=window_length,
    window_overlap=window_overlap,
    phase_method=PhaseMethod.HILBERT,
    sync_method=SyncMethod.WPLI,
    sfreq=256.0
)

# For CSD method (requires additional parameters)
sync_matrix = synchronization(
    file_path=file_path,
    lbands=lbands,
    hbands=hbands,
    window_length=window_length,  # epochs per window
    window_overlap=7,             # shared epochs
    samples_per_epoch=1280,
    epochs_overlap=0.5,
    phase_method=PhaseMethod.CSD,
    sync_method=SyncMethod.WPLI,
    sfreq=256.0
)

print(f"Result shape: {sync_matrix.shape}")
# Output: (n_bands, n_windows, n_channels, n_channels)
```

### Dataset Processing

```python
from utils import process_eeg_files, find_seizures_times, create_header

# Find seizure information
seizures = find_seizures_times('path/to/chb01/')

# Create processing header
header = create_header(
    phase_method='hilbert',
    sync_method='wpli',
    window_length=1280,
    window_overlap=0.5,
    lbands=[8, 13, 30],
    hbands=[12, 30, 100]
)

# Define processing function
def process_file(file_path):
    return synchronization(
        file_path=file_path,
        lbands=[8, 13, 30],
        hbands=[12, 30, 100],
        window_length=1280,
        window_overlap=0.5,
        phase_method='hilbert',
        sync_method='wpli'
    )

# Process all files in directory
results = process_eeg_files(
    dir_path='path/to/chb01/',
    process_func=process_file,
    cache_name='wpli_analysis',
    header=header,
    verbose=True,
    max_workers=4
)
```

## Phase Methods

### 1. Cross-Spectral Density (CSD)
- Uses Welch's method for robust spectral estimation
- Requires epoching of data
- Better for noisy signals
- Parameters: `samples_per_epoch`, `epochs_overlap`, `epochs_per_window`

### 2. Hilbert Transform
- Direct instantaneous phase extraction
- Faster computation
- Good for clean signals
- Parameters: `window_length`, `window_overlap`

## Synchronization Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **WPLI** | Weighted Phase Lag Index | [0, 1] |
| **PLI** | Phase Lag Index | [0, 1] |



## Dataset Support

### CHB-MIT Database
```python
from utils import find_seizures_times, find_records_times, find_electrodes_placements

# Automatically detects CHB format from path
seizures = find_seizures_times('path/to/chb01/')
records = find_records_times('path/to/chb01/')
electrodes = find_electrodes_placements('path/to/chb01/')
```

### Private Dataset
```python
# Supports custom EDF datasets with annotation files
seizures = find_seizures_times('path/to/Pat001/')
records = find_records_times('path/to/Pat001/')
```

## Output Format

The synchronization functions return arrays with shape:
- **CSD Method**: `(n_bands, n_windows, n_channels, n_channels)`
- **Hilbert Method**: `(n_bands, n_windows, n_channels, n_channels)`

Where:
- `n_bands`: Number of frequency bands
- `n_windows`: Number of time windows
- `n_channels`: Number of EEG channels

## Performance Optimization

- **Numba JIT**: Automatic acceleration for computational bottlenecks
- **Parallel Processing**: Multi-threaded band processing
- **Memory Monitoring**: Prevents system overload
- **Caching**: SQLite-based result storage
- **Optimized Algorithms**: Vectorized operations where possible

## Example Workflows

### Seizure Analysis
```python
# 1. Load seizure times
seizures = find_seizures_times('path/to/patient/')

# 2. Process EEG files
results = process_eeg_files('path/to/patient/', process_func)

# 3. Analyze connectivity around seizure events
for filename, seizure_list in seizures.items():
    if seizure_list:  # Has seizures
        connectivity = results[filename]
        # Analyze pre/post seizure connectivity changes
```

### Multi-Band Analysis
```python
# Define multiple frequency bands
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

lbands = [band[0] for band in bands.values()]
hbands = [band[1] for band in bands.values()]

# Process with all bands
sync_matrix = synchronization(
    file_path=file_path,
    lbands=lbands,
    hbands=hbands,
    # ... other parameters
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `python utils/tests.py`
5. Submit a pull request

## Support

For questions or issues:
- Check the documentation in source files
- Review example usage in test files
- Ensure all dependencies are correctly installed
- Verify EEG file formats are supported (.edf)