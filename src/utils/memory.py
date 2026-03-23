"""Utility functions for tracking memory usage (CPU + GPU)."""

import os
import psutil
import torch


def get_ram_usage_mb() -> float:
    """
    Get current RAM usage of the process in MB.
    """
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 ** 2)


def reset_peak_gpu_memory():
    """
    Reset PyTorch peak GPU memory stats.
    Safe to call even if CUDA is not available.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_peak_gpu_memory_mb() -> float:
    """
    Get peak GPU memory usage in MB.
    Returns 0.0 if CUDA is not available.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0