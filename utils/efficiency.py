"""Runtime efficiency helpers."""
from __future__ import annotations

import resource
import sys

import torch


def load_model_with_dtype(loader, model_path, dtype, **kwargs):
    """Load a model, falling back from ``dtype=`` to ``torch_dtype=`` kwarg."""
    try:
        return loader(model_path, dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return loader(model_path, torch_dtype=dtype, **kwargs)


def reset_gpu_peak_memory(device=None):
    """Reset CUDA peak-memory tracking counters."""
    if not torch.cuda.is_available():
        return
    if device is not None and device.type != "cuda":
        return
    if device is None or device.index is None:
        torch.cuda.reset_peak_memory_stats()
    else:
        torch.cuda.reset_peak_memory_stats(device.index)


def get_gpu_peak_memory_gb(device=None):
    """Return peak GPU memory allocation in GiB."""
    if not torch.cuda.is_available():
        return 0.0
    if device is not None and device.type != "cuda":
        return 0.0
    if device is None or device.index is None:
        peak_bytes = torch.cuda.max_memory_allocated()
    else:
        peak_bytes = torch.cuda.max_memory_allocated(device.index)
    return float(peak_bytes) / (1024 ** 3)


def get_cpu_peak_memory_gb():
    """Return peak RSS in GiB (platform-aware)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(peak) / (1024 ** 3)
    return float(peak) / (1024 ** 2)
