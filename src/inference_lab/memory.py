import torch


def reset_peak_stats() -> None:
    torch.cuda.reset_peak_memory_stats()


def peak_allocated_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


def peak_reserved_mb() -> float:
    return torch.cuda.max_memory_reserved() / 1024**2


def current_allocated_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2
