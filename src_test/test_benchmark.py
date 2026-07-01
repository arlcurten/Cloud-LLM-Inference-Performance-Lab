import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference_lab.config import InferenceConfig
from inference_lab.benchmark import validate_benchmark_config
from inference_lab.metrics import compute_stats, aggregate_iterations

# ── compute_stats ────────────────────────────────────────────────────────────

def test_compute_stats_basic():
    stats = compute_stats([10.0, 20.0, 30.0, 40.0, 50.0])
    assert stats["mean"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0


def test_compute_stats_median_odd():
    stats = compute_stats([1.0, 2.0, 3.0])
    assert stats["median"] == 2.0


def test_compute_stats_median_even():
    stats = compute_stats([1.0, 2.0, 3.0, 4.0])
    assert stats["median"] == 2.5


def test_compute_stats_empty_raises():
    with pytest.raises(ValueError):
        compute_stats([])


# ── percentile calculation ────────────────────────────────────────────────────

def test_p95_small_sample():
    # With n=10, int(10*0.95)=9, so p95 == max
    values = list(range(1, 11))          # [1..10]
    stats = compute_stats(values)
    assert stats["p95"] == stats["max"]


def test_p95_large_sample():
    # With n=100, p95 index = 95 → value 96 (0-indexed sorted list of 1..100)
    values = list(range(1, 101))
    stats = compute_stats(values)
    assert stats["p95"] == 96.0


def test_p95_single_value():
    stats = compute_stats([42.0])
    assert stats["p95"] == 42.0
    assert stats["mean"] == 42.0
    assert stats["min"] == stats["max"] == 42.0


# ── aggregate_iterations ─────────────────────────────────────────────────────

def _make_fake_iteration(i: int) -> dict:
    base = float(i + 1)
    return {
        "iteration": i,
        "input_token_count": 12,
        "generated_token_count": 100,
        "prefill_latency_ms": base * 10,
        "decode_total_latency_ms": base * 100,
        "decode_mean_token_latency_ms": base * 1.5,
        "decode_median_token_latency_ms": base * 1.4,
        "decode_p95_token_latency_ms": base * 1.8,
        "decode_tokens_per_second": 70.0 / base,
        "e2e_latency_ms": base * 110,
        "peak_allocated_mb": 1920.0,
        "peak_reserved_mb": 1950.0,
    }


def test_aggregate_has_required_keys():
    iterations = [_make_fake_iteration(i) for i in range(5)]
    summary = aggregate_iterations(iterations)
    required_stat_keys = [
        "prefill_latency_ms",
        "decode_total_latency_ms",
        "decode_mean_token_latency_ms",
        "decode_median_token_latency_ms",
        "decode_p95_token_latency_ms",
        "decode_tokens_per_second",
        "e2e_latency_ms",
        "peak_allocated_mb",
        "peak_reserved_mb",
    ]
    for key in required_stat_keys:
        assert key in summary, f"Missing summary key: {key}"
        for stat in ("mean", "median", "p95", "min", "max"):
            assert stat in summary[key], f"Missing stat '{stat}' in {key}"
    assert "input_token_count" in summary
    assert "generated_token_count" in summary


def test_aggregate_passthrough_token_counts():
    iterations = [_make_fake_iteration(i) for i in range(3)]
    summary = aggregate_iterations(iterations)
    assert summary["input_token_count"] == 12
    assert summary["generated_token_count"] == 100


# ── required JSON result fields from a single iteration ──────────────────────

REQUIRED_ITERATION_FIELDS = [
    "input_token_count",
    "generated_token_count",
    "prefill_latency_ms",
    "decode_total_latency_ms",
    "decode_mean_token_latency_ms",
    "decode_median_token_latency_ms",
    "decode_p95_token_latency_ms",
    "decode_tokens_per_second",
    "e2e_latency_ms",
    "peak_allocated_mb",
    "peak_reserved_mb",
]

REQUIRED_JSON_TOP_KEYS = ["metadata", "configuration", "iterations", "summary"]

REQUIRED_METADATA_KEYS = [
    "timestamp", "git_commit", "model_id", "dtype", "device",
    "gpu_name", "total_vram_mb", "python_version", "pytorch_version",
    "transformers_version", "cuda_version",
]

REQUIRED_CONFIG_KEYS = [
    "prompt", "max_new_tokens", "warmup_iterations",
    "measurement_iterations", "do_sample", "seed",
]


def test_iteration_result_has_required_fields():
    fake = _make_fake_iteration(0)
    for field in REQUIRED_ITERATION_FIELDS:
        assert field in fake, f"Missing iteration field: {field}"


def test_json_top_level_keys():
    """The assembled result document must have all required top-level keys."""
    doc = {
        "metadata": {},
        "configuration": {},
        "iterations": [],
        "summary": {},
    }
    for key in REQUIRED_JSON_TOP_KEYS:
        assert key in doc


def test_metadata_keys():
    meta = {k: "x" for k in REQUIRED_METADATA_KEYS}
    for key in REQUIRED_METADATA_KEYS:
        assert key in meta


def test_configuration_keys():
    conf = {k: None for k in REQUIRED_CONFIG_KEYS}
    for key in REQUIRED_CONFIG_KEYS:
        assert key in conf


# ── validate_benchmark_config ─────────────────────────────────────────────────

def test_valid_config_passes():
    cfg = InferenceConfig(measurement_iterations=10, warmup_iterations=3)
    validate_benchmark_config(cfg)  # should not raise


def test_measurement_iterations_none_raises():
    cfg = InferenceConfig(measurement_iterations=None)
    with pytest.raises(ValueError, match="measurement_iterations"):
        validate_benchmark_config(cfg)


def test_measurement_iterations_zero_raises():
    cfg = InferenceConfig(measurement_iterations=0)
    with pytest.raises(ValueError, match="measurement_iterations"):
        validate_benchmark_config(cfg)


def test_measurement_iterations_negative_raises():
    cfg = InferenceConfig(measurement_iterations=-1)
    with pytest.raises(ValueError, match="measurement_iterations"):
        validate_benchmark_config(cfg)


def test_warmup_iterations_negative_raises():
    cfg = InferenceConfig(measurement_iterations=5, warmup_iterations=-1)
    with pytest.raises(ValueError, match="warmup_iterations"):
        validate_benchmark_config(cfg)


def test_warmup_iterations_zero_is_valid():
    cfg = InferenceConfig(measurement_iterations=5, warmup_iterations=0)
    validate_benchmark_config(cfg)  # should not raise
