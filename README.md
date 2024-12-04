# MUFASA

A Python utility module for CPU core management and GPU memory optimization, particularly useful for machine learning workflows.

## Installation

You can install MUFASA directly from PyPI:

```bash
pip install mufasa-polimi
```

## Features

- CPU core detection and optimization for SLURM environments
- Automated GPU memory management and cleanup
- Detailed memory usage reporting

## Usage

### Core Management Functions

```python
from mufasa import getCoreAffinity, setOptimalWorkers

# Get available CPU cores
cpu_count = getCoreAffinity()
print(f"Available CPU cores: {cpu_count}")

# Set optimal number of worker processes
workers = setOptimalWorkers()
print(f"Optimal worker count: {workers}")
```

#### `getCoreAffinity()`
Detects the number of available CPU cores, taking into account SLURM job allocations if running in a SLURM environment. Returns the minimum between SLURM-allocated CPUs and system-available CPUs, or the total system CPU count if not in a SLURM environment.

#### `setOptimalWorkers()`
Similar to `getCoreAffinity()`, but defaults to 1 if no SLURM environment is detected. Useful for setting worker counts in parallel processing scenarios.

### GPU Memory Management

```python
from mufasa import gpuClean

# Basic cleanup
freed_count, freed_memory = gpuClean()

# Detailed cleanup with verbose output
freed_count, freed_memory = gpuClean(
    exclude_vars=['model', 'optimizer'],  # Variables to preserve
    verbose=True  # Enable detailed reporting
)
```

#### `gpuClean(local_vars=None, exclude_vars=None, verbose=False)`

Automatically detects and frees GPU memory by cleaning up tensor variables.

**Parameters:**
- `local_vars` (dict, optional): Dictionary of local variables to clean. If None, uses the calling frame's locals.
- `exclude_vars` (list, optional): List of variable names to exclude from cleanup.
- `verbose` (bool): Whether to print detailed information about cleaned variables.

**Returns:**
- tuple: (freed_count, freed_memory_mb)
  - freed_count: Number of tensors freed
  - freed_memory_mb: Approximate memory freed in MB

**Features:**
- Cleans up PyTorch tensors in local scope
- Handles nested tensors in dictionaries and lists
- Provides detailed memory usage reports when verbose=True
- Allows excluding specific variables from cleanup
- Automatically triggers garbage collection and GPU memory cache clearing

**Example with Verbose Output:**
```python
import torch
from mufasa import gpuClean

# Create some example tensors
tensor1 = torch.randn(1000, 1000).cuda()
tensor2 = torch.randn(2000, 2000).cuda()

# Clean up with detailed output
freed_count, freed_memory = gpuClean(verbose=True)
```

The verbose output includes:
- Table of cleaned tensors with their shapes and sizes
- Total number of tensors freed
- Total memory freed
- Current GPU memory allocation status
- List of excluded variables (if any)

## Notes

- SLURM-specific features require a SLURM environment
- GPU cleaning functions require PyTorch and a CUDA-capable GPU
- Memory sizes are reported in MB or GB depending on the size
- The module uses the `rich` library for formatted console output in verbose mode