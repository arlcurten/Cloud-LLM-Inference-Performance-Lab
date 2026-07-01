"""Tests for aggregate_results.py and plot_results.py (no model required)."""
import csv
import json
import sys
from io import StringIO
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from aggregate_results import _extract_row, aggregate, collect_json_files, CSV_COLUMNS
from plot_results import validate_csv_columns, REQUIRED_PLOT_COLS


# ── Sample data ──────────────────────────────────────────────────────────────

def _make_sample_doc(experiment_name="phase1_short", n_iters=3):
    """Build a minimal well-formed benchmark JSON document."""
    iteration = {
        "iteration": 0,
        "input_tokens": 30,
        "generated_tokens": 32,
        "prefill_latency_ms": 50.0,
        "decode_total_latency_ms": 800.0,
        "decode_token_latencies_ms": [25.0] * 31,
        "mean_decode_token_latency_ms": 25.0,
        "median_decode_token_latency_ms": 25.0,
        "p95_decode_token_latency_ms": 28.0,
        "decode_tokens_per_second": 38.5,
        "e2e_latency_ms": 850.0,
        "peak_cuda_allocated_mb": 1922.2,
        "peak_cuda_reserved_mb": 1944.0,
    }
    iterations = [{**iteration, "iteration": i} for i in range(n_iters)]

    summary = {
        "prefill_latency_ms": {"mean": 50.0, "median": 50.0, "p95": 50.0, "min": 50.0, "max": 50.0},
        "decode_total_latency_ms": {"mean": 800.0, "median": 800.0, "p95": 800.0, "min": 800.0, "max": 800.0},
        "mean_decode_token_latency_ms": {"mean": 25.0, "median": 25.0, "p95": 25.0, "min": 25.0, "max": 25.0},
        "median_decode_token_latency_ms": {"mean": 25.0, "median": 25.0, "p95": 25.0, "min": 25.0, "max": 25.0},
        "p95_decode_token_latency_ms": {"mean": 28.0, "median": 28.0, "p95": 28.0, "min": 28.0, "max": 28.0},
        "decode_tokens_per_second": {"mean": 38.5, "median": 38.5, "p95": 38.5, "min": 38.5, "max": 38.5},
        "e2e_latency_ms": {"mean": 850.0, "median": 850.0, "p95": 850.0, "min": 850.0, "max": 850.0},
        "peak_cuda_allocated_mb": {"mean": 1922.2, "median": 1922.2, "p95": 1922.2, "min": 1922.2, "max": 1922.2},
        "peak_cuda_reserved_mb": {"mean": 1944.0, "median": 1944.0, "p95": 1944.0, "min": 1944.0, "max": 1944.0},
        "generated_tokens": {"mean": 32.0, "median": 32.0, "p95": 32.0, "min": 32.0, "max": 32.0},
        "input_tokens": 30,
    }

    return {
        "metadata": {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "experiment_name": experiment_name,
            "git_commit": "abc1234",
            "model_id": "google/gemma-3-1b-it",
            "dtype": "float16",
            "device": "cuda",
            "gpu_name": "NVIDIA Test GPU",
            "total_vram_mb": 4096.0,
            "python_version": "3.12.0",
            "pytorch_version": "2.0.0",
            "transformers_version": "5.0.0",
            "cuda_version": "12.0",
        },
        "configuration": {
            "prompt": "test prompt",
            "max_new_tokens": 32,
            "warmup_iterations": 3,
            "measurement_iterations": n_iters,
            "do_sample": False,
            "seed": 42,
        },
        "iterations": iterations,
        "summary": summary,
    }


# ── aggregation from sample JSON data ────────────────────────────────────────

def test_extract_row_returns_all_csv_columns():
    doc = _make_sample_doc()
    row = _extract_row(doc, Path("benchmark_phase1_short_20260101_000000.json"))
    for col in CSV_COLUMNS:
        assert col in row, f"Missing CSV column: {col}"


def test_extract_row_values():
    doc = _make_sample_doc("phase1_short")
    row = _extract_row(doc, Path("benchmark_phase1_short_20260101_000000.json"))
    assert row["experiment_name"] == "phase1_short"
    assert row["model_id"] == "google/gemma-3-1b-it"
    assert row["dtype"] == "float16"
    assert row["input_tokens"] == 30
    assert row["requested_output_tokens"] == 32
    assert float(row["mean_generated_tokens"]) == 32.0
    assert float(row["mean_prefill_latency_ms"]) == 50.0
    assert float(row["mean_decode_tokens_per_second"]) == 38.5
    assert float(row["mean_e2e_latency_ms"]) == 850.0
    assert float(row["peak_cuda_allocated_mb"]) == 1922.2
    assert float(row["peak_cuda_reserved_mb"]) == 1944.0


def test_extract_row_falls_back_to_filename_for_experiment_name(tmp_path):
    doc = _make_sample_doc()
    del doc["metadata"]["experiment_name"]
    path = tmp_path / "benchmark_my_run_123.json"
    row = _extract_row(doc, path)
    assert row["experiment_name"] == "benchmark_my_run_123"


def test_aggregate_multiple_docs(tmp_path):
    for name in ("phase1_short", "phase1_medium"):
        doc = _make_sample_doc(name)
        (tmp_path / f"{name}.json").write_text(json.dumps(doc))

    files = list(tmp_path.glob("*.json"))
    rows = aggregate(files)
    assert len(rows) == 2
    names = {r["experiment_name"] for r in rows}
    assert names == {"phase1_short", "phase1_medium"}


def test_aggregate_skips_bad_file(tmp_path):
    (tmp_path / "valid.json").write_text(json.dumps(_make_sample_doc("good")))
    (tmp_path / "bad.json").write_text("not valid json {{")
    files = sorted(tmp_path.glob("*.json"))
    rows = aggregate(files)
    assert len(rows) == 1
    assert rows[0]["experiment_name"] == "good"


# ── required CSV columns ──────────────────────────────────────────────────────

def test_csv_columns_list_complete():
    required = [
        "experiment_name", "model_id", "dtype",
        "input_tokens", "requested_output_tokens", "mean_generated_tokens",
        "mean_prefill_latency_ms", "median_prefill_latency_ms",
        "mean_decode_token_latency_ms", "median_decode_token_latency_ms",
        "mean_decode_tokens_per_second",
        "mean_e2e_latency_ms", "median_e2e_latency_ms",
        "peak_cuda_allocated_mb", "peak_cuda_reserved_mb",
    ]
    for col in required:
        assert col in CSV_COLUMNS, f"CSV_COLUMNS missing: {col}"


# ── plot input validation ─────────────────────────────────────────────────────

def test_validate_csv_columns_passes_with_all_columns():
    headers = REQUIRED_PLOT_COLS + ["extra_col"]
    validate_csv_columns(headers)  # should not raise


def test_validate_csv_columns_raises_on_missing():
    headers = [c for c in REQUIRED_PLOT_COLS if c != "mean_prefill_latency_ms"]
    with pytest.raises(ValueError, match="mean_prefill_latency_ms"):
        validate_csv_columns(headers)


def test_validate_csv_columns_raises_on_empty():
    with pytest.raises(ValueError):
        validate_csv_columns([])


# ── metric schema completeness ────────────────────────────────────────────────

REQUIRED_SUMMARY_STAT_KEYS = [
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


def test_summary_schema_completeness():
    doc = _make_sample_doc()
    for key in REQUIRED_SUMMARY_STAT_KEYS:
        assert key in doc["summary"], f"Summary missing: {key}"
        assert isinstance(doc["summary"][key], dict)
        for stat in ("mean", "median", "p95", "min", "max"):
            assert stat in doc["summary"][key]
    assert "input_tokens" in doc["summary"]


def test_iteration_schema_completeness():
    doc = _make_sample_doc()
    it = doc["iterations"][0]
    required_fields = [
        "input_tokens", "generated_tokens",
        "prefill_latency_ms", "decode_total_latency_ms",
        "decode_token_latencies_ms",
        "mean_decode_token_latency_ms", "median_decode_token_latency_ms",
        "p95_decode_token_latency_ms", "decode_tokens_per_second",
        "e2e_latency_ms", "peak_cuda_allocated_mb", "peak_cuda_reserved_mb",
    ]
    for field in required_fields:
        assert field in it, f"Iteration missing: {field}"
    assert isinstance(it["decode_token_latencies_ms"], list)
