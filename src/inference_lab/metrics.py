import statistics
from typing import Sequence

# Scalar metrics that are aggregated (mean/median/p95/min/max) across iterations.
# decode_token_latencies_ms is a list-per-iteration and is excluded from aggregation.
_AGGREGATE_KEYS = [
    "prefill_latency_ms",
    "decode_total_latency_ms",
    "mean_decode_token_latency_ms",
    "median_decode_token_latency_ms",
    "p95_decode_token_latency_ms",
    "decode_tokens_per_second",
    "e2e_latency_ms",
    "peak_cuda_allocated_mb",
    "peak_cuda_reserved_mb",
    "generated_tokens",
]


def compute_stats(values: Sequence[float]) -> dict:
    """Return mean/median/p95/min/max for a sequence of floats."""
    if not values:
        raise ValueError("Cannot compute stats on an empty sequence.")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    p95_idx = min(int(n * 0.95), n - 1)
    return {
        "mean": round(statistics.mean(sorted_vals), 3),
        "median": round(statistics.median(sorted_vals), 3),
        "p95": round(sorted_vals[p95_idx], 3),
        "min": round(sorted_vals[0], 3),
        "max": round(sorted_vals[-1], 3),
    }


def aggregate_iterations(iterations: list[dict]) -> dict:
    """Compute per-metric stats across all measured iterations."""
    summary = {}
    for key in _AGGREGATE_KEYS:
        values = [it[key] for it in iterations]
        summary[key] = compute_stats(values)
    # input_tokens is constant (it is the input, not generated output).
    summary["input_tokens"] = iterations[0]["input_tokens"]
    return summary
