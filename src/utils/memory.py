import os
import psutil


def get_ram_usage_mb() -> float:
    """
    Return the current RAM usage of the running Python process.

    This measures the resident memory (RSS) used by the experiment.

    Returns:
        float: memory usage in megabytes
    """
    process = psutil.Process(os.getpid())
    memory_bytes = process.memory_info().rss
    memory_mb = memory_bytes / (1024 ** 2)
    return memory_mb