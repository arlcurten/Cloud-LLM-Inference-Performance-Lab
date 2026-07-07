"""Tests for Phase 2B online_benchmark.py (no GPU/server required)."""
import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from inference_lab.online_benchmark import (
    ConcurrencyBenchmarkConfig,
    RequestResult,
    aggregate_concurrency_results,
    compute_approx_tpot_ms,
    compute_e2e_latency_ms,
    compute_output_tokens_per_second,
    compute_ttft_ms,
    load_concurrency_config,
    run_closed_loop,
    run_single_request,
    validate_concurrency_config,
)
from inference_lab import online_benchmark

import aggregate_concurrency_results as agg_cli


# ── TTFT / E2E / approx TPOT / throughput calculations ───────────────────────

def test_compute_ttft_ms():
    assert compute_ttft_ms(100.0, 100.05) == pytest.approx(50.0)


def test_compute_e2e_latency_ms():
    assert compute_e2e_latency_ms(100.0, 100.3) == pytest.approx(300.0)


def test_compute_approx_tpot_ms_basic():
    # e2e=500ms, ttft=100ms, 5 completion tokens -> (500-100)/(5-1) = 100ms
    assert compute_approx_tpot_ms(500.0, 100.0, 5) == pytest.approx(100.0)


def test_compute_approx_tpot_ms_none_when_single_token():
    assert compute_approx_tpot_ms(500.0, 100.0, 1) is None


def test_compute_approx_tpot_ms_none_when_no_tokens():
    assert compute_approx_tpot_ms(500.0, 100.0, None) is None


def test_compute_approx_tpot_ms_none_when_ttft_missing():
    assert compute_approx_tpot_ms(500.0, None, 5) is None


def test_compute_output_tokens_per_second():
    assert compute_output_tokens_per_second(10, 2000.0) == pytest.approx(5.0)


def test_compute_output_tokens_per_second_none_when_no_tokens():
    assert compute_output_tokens_per_second(None, 2000.0) is None


def test_compute_output_tokens_per_second_none_when_zero_latency():
    assert compute_output_tokens_per_second(10, 0.0) is None


# ── ConcurrencyBenchmarkConfig ────────────────────────────────────────────────

def test_load_concurrency_config_from_yaml(tmp_path):
    yaml_path = tmp_path / "concurrency.yaml"
    yaml_path.write_text(
        "server_url: http://127.0.0.1:8000\n"
        "model_id: models/google/gemma-3-1b-it\n"
        "prompt: hello\n"
        "max_tokens: 32\n"
        "temperature: 0\n"
        "stream: true\n"
        "concurrency_levels: [1, 2, 4]\n"
        "requests_per_level: 30\n"
        "request_timeout_seconds: 30\n"
        "warmup_requests: 3\n"
        "seed: 42\n"
    )
    cfg = load_concurrency_config(yaml_path)
    assert cfg.concurrency_levels == [1, 2, 4]
    assert cfg.requests_per_level == 30


def test_validate_concurrency_config_passes():
    cfg = ConcurrencyBenchmarkConfig(server_url="http://x", model_id="m", prompt="p")
    validate_concurrency_config(cfg)  # should not raise


def test_validate_concurrency_config_rejects_non_streaming():
    cfg = ConcurrencyBenchmarkConfig(server_url="http://x", model_id="m", prompt="p", stream=False)
    with pytest.raises(ValueError, match="stream"):
        validate_concurrency_config(cfg)


def test_validate_concurrency_config_rejects_empty_levels():
    cfg = ConcurrencyBenchmarkConfig(server_url="http://x", model_id="m", prompt="p", concurrency_levels=[])
    with pytest.raises(ValueError, match="concurrency_levels"):
        validate_concurrency_config(cfg)


def test_validate_concurrency_config_rejects_negative_level():
    cfg = ConcurrencyBenchmarkConfig(server_url="http://x", model_id="m", prompt="p", concurrency_levels=[1, -2])
    with pytest.raises(ValueError, match="concurrency_levels"):
        validate_concurrency_config(cfg)


def test_validate_concurrency_config_rejects_zero_requests_per_level():
    cfg = ConcurrencyBenchmarkConfig(server_url="http://x", model_id="m", prompt="p", requests_per_level=0)
    with pytest.raises(ValueError, match="requests_per_level"):
        validate_concurrency_config(cfg)


# ── RequestResult schema ──────────────────────────────────────────────────────

def test_request_result_to_dict_has_required_keys():
    r = RequestResult(
        request_start_time=0.0,
        first_meaningful_chunk_time=0.1,
        request_end_time=0.5,
        ttft_ms=100.0,
        e2e_latency_ms=500.0,
        prompt_tokens=5,
        completion_tokens=10,
        success=True,
        http_status=200,
        error_type=None,
    )
    d = r.to_dict()
    required = [
        "request_start_time", "first_meaningful_chunk_time", "request_end_time",
        "ttft_ms", "e2e_latency_ms", "prompt_tokens", "completion_tokens",
        "success", "http_status", "error_type",
    ]
    for key in required:
        assert key in d, f"Missing RequestResult field: {key}"


# ── closed-loop concurrency enforcement ───────────────────────────────────────

@pytest.mark.asyncio
async def test_run_closed_loop_never_exceeds_concurrency():
    active = 0
    max_active = 0
    tracker_lock = asyncio.Lock()

    async def request_fn():
        nonlocal active, max_active
        async with tracker_lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        async with tracker_lock:
            active -= 1
        return {"success": True}

    results = await run_closed_loop(request_fn, concurrency=3, num_requests=12)
    assert len(results) == 12
    assert max_active <= 3
    assert max_active == 3


@pytest.mark.asyncio
async def test_run_closed_loop_concurrency_one_is_fully_sequential():
    active = 0
    max_active = 0
    tracker_lock = asyncio.Lock()

    async def request_fn():
        nonlocal active, max_active
        async with tracker_lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.005)
        async with tracker_lock:
            active -= 1
        return {"success": True}

    results = await run_closed_loop(request_fn, concurrency=1, num_requests=5)
    assert len(results) == 5
    assert max_active == 1


@pytest.mark.asyncio
async def test_run_closed_loop_rejects_non_positive_concurrency():
    async def request_fn():
        return {"success": True}

    with pytest.raises(ValueError, match="concurrency"):
        await run_closed_loop(request_fn, concurrency=0, num_requests=5)


@pytest.mark.asyncio
async def test_run_closed_loop_rejects_non_positive_num_requests():
    async def request_fn():
        return {"success": True}

    with pytest.raises(ValueError, match="num_requests"):
        await run_closed_loop(request_fn, concurrency=2, num_requests=0)


# ── run_single_request against a mocked streaming client ────────────────────

class _FakeStreamResponse:
    def __init__(self, lines, status_code=200, raise_exc=None):
        self._lines = lines
        self.status_code = status_code
        self._raise_exc = raise_exc

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc


class _FakeStreamCM:
    def __init__(self, response=None, raise_on_enter=None):
        self._response = response
        self._raise_on_enter = raise_on_enter

    async def __aenter__(self):
        if self._raise_on_enter:
            raise self._raise_on_enter
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, cm):
        self._cm = cm

    def stream(self, method, url, json=None, timeout=None):
        return self._cm


def _fake_perf_counter_sequence(values):
    it = iter(values)

    def _fake():
        return next(it)

    return _fake


@pytest.mark.asyncio
async def test_run_single_request_ignores_empty_chunks(monkeypatch):
    lines = [
        'data: {"choices": [{"text": ""}]}',
        'data: {"choices": [{"text": "Hello"}]}',
        'data: {"choices": [{"text": " world"}]}',
        'data: {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}',
        "data: [DONE]",
    ]
    client = _FakeClient(_FakeStreamCM(_FakeStreamResponse(lines)))
    # perf_counter calls: request_start, first-meaningful-chunk (only once), request_end
    monkeypatch.setattr(online_benchmark.time, "perf_counter", _fake_perf_counter_sequence([0.0, 0.05, 0.30]))

    result = await run_single_request(client, "http://x", "m", "prompt", 32, 0.0, 30.0)

    assert result.success is True
    assert result.ttft_ms == pytest.approx(50.0)
    assert result.e2e_latency_ms == pytest.approx(300.0)
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 2
    assert result.error_type is None


@pytest.mark.asyncio
async def test_run_single_request_no_content_is_failure():
    # Only an empty/metadata chunk arrives -> no meaningful token ever received.
    lines = [
        'data: {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 0}}',
        "data: [DONE]",
    ]
    client = _FakeClient(_FakeStreamCM(_FakeStreamResponse(lines)))
    result = await run_single_request(client, "http://x", "m", "prompt", 32, 0.0, 30.0)
    assert result.success is False
    assert result.error_type == "no_content_received"
    assert result.ttft_ms is None


@pytest.mark.asyncio
async def test_run_single_request_timeout_captured():
    client = _FakeClient(_FakeStreamCM(raise_on_enter=httpx.TimeoutException("timed out")))
    result = await run_single_request(client, "http://x", "m", "prompt", 32, 0.0, 30.0)
    assert result.success is False
    assert result.error_type == "timeout"


@pytest.mark.asyncio
async def test_run_single_request_http_error_captured():
    request = httpx.Request("POST", "http://x/v1/completions")
    response = httpx.Response(500, request=request)
    error = httpx.HTTPStatusError("server error", request=request, response=response)
    fake_response = _FakeStreamResponse([], status_code=500, raise_exc=error)
    client = _FakeClient(_FakeStreamCM(fake_response))
    result = await run_single_request(client, "http://x", "m", "prompt", 32, 0.0, 30.0)
    assert result.success is False
    assert result.error_type == "http_500"


@pytest.mark.asyncio
async def test_run_single_request_malformed_json_captured():
    lines = ["data: {not valid json"]
    client = _FakeClient(_FakeStreamCM(_FakeStreamResponse(lines)))
    result = await run_single_request(client, "http://x", "m", "prompt", 32, 0.0, 30.0)
    assert result.success is False
    assert result.error_type is not None


# ── aggregation ────────────────────────────────────────────────────────────────

def _make_result(ttft_ms, e2e_ms, completion_tokens, success=True, error_type=None):
    return RequestResult(
        request_start_time=0.0,
        first_meaningful_chunk_time=ttft_ms / 1000 if ttft_ms is not None else None,
        request_end_time=e2e_ms / 1000,
        ttft_ms=ttft_ms,
        e2e_latency_ms=e2e_ms,
        prompt_tokens=8,
        completion_tokens=completion_tokens,
        success=success,
        http_status=200 if success else 500,
        error_type=error_type,
    )


def test_aggregate_request_throughput():
    results = [_make_result(50.0, 500.0, 10) for _ in range(10)]
    summary = aggregate_concurrency_results(results, duration_seconds=5.0)
    assert summary["success_count"] == 10
    assert summary["request_throughput_rps"] == pytest.approx(2.0)


def test_aggregate_token_throughput_uses_total_not_average():
    # 10 requests x 10 completion tokens = 100 tokens over 5s -> 20 tok/s aggregate.
    # This must not equal an average of per-request token throughputs.
    results = [_make_result(50.0, 500.0, 10) for _ in range(10)]
    summary = aggregate_concurrency_results(results, duration_seconds=5.0)
    assert summary["total_completion_tokens"] == 100
    assert summary["output_token_throughput_tps"] == pytest.approx(20.0)


def test_aggregate_failure_rate():
    results = [_make_result(50.0, 500.0, 10) for _ in range(7)]
    results += [_make_result(None, 1000.0, None, success=False, error_type="timeout") for _ in range(3)]
    summary = aggregate_concurrency_results(results, duration_seconds=10.0)
    assert summary["request_count"] == 10
    assert summary["success_count"] == 7
    assert summary["failure_count"] == 3
    assert summary["failure_rate"] == pytest.approx(0.3)


def test_aggregate_percentiles():
    # ttft values 1..10 ms -> p95 index = int(10*0.95)=9 -> value 10; p99 same clamp -> 10
    results = [_make_result(float(i), float(i) + 100.0, 5) for i in range(1, 11)]
    summary = aggregate_concurrency_results(results, duration_seconds=1.0)
    assert summary["ttft_p95_ms"] == pytest.approx(10.0)
    assert summary["ttft_p99_ms"] == pytest.approx(10.0)
    assert summary["ttft_median_ms"] == pytest.approx(5.5)


def test_aggregate_approx_tpot_stats():
    # e2e - ttft = 100ms with completion_tokens=5 -> tpot = 100/4 = 25ms for all
    results = [_make_result(50.0, 150.0, 5) for _ in range(5)]
    summary = aggregate_concurrency_results(results, duration_seconds=1.0)
    assert summary["approx_tpot_mean_ms"] == pytest.approx(25.0)
    assert summary["approx_tpot_median_ms"] == pytest.approx(25.0)


def test_aggregate_excludes_failures_from_latency_stats():
    results = [_make_result(50.0, 500.0, 10) for _ in range(3)]
    results += [_make_result(None, 30000.0, None, success=False, error_type="timeout")]
    summary = aggregate_concurrency_results(results, duration_seconds=10.0)
    assert summary["e2e_mean_ms"] == pytest.approx(500.0)  # failures excluded


def test_aggregate_rejects_empty_results():
    with pytest.raises(ValueError):
        aggregate_concurrency_results([], duration_seconds=1.0)


# ── CSV column completeness ───────────────────────────────────────────────────

def _make_sample_concurrency_doc(concurrency_levels=(1, 2)):
    summary_keys = [
        "request_count", "success_count", "failure_count", "failure_rate",
        "ttft_mean_ms", "ttft_median_ms", "ttft_p95_ms", "ttft_p99_ms",
        "approx_tpot_mean_ms", "approx_tpot_median_ms", "approx_tpot_p95_ms",
        "e2e_mean_ms", "e2e_median_ms", "e2e_p95_ms", "e2e_p99_ms",
        "request_throughput_rps", "output_token_throughput_tps",
        "total_completion_tokens", "mean_completion_tokens",
        "benchmark_duration_seconds",
    ]
    summary = {k: 1.0 for k in summary_keys}
    return {
        "metadata": {"timestamp": "2026-01-01T00:00:00+00:00"},
        "configuration": {"prompt": "p", "max_tokens": 32},
        "experiments": [
            {"concurrency": c, "raw_requests": [], "summary": dict(summary)}
            for c in concurrency_levels
        ],
    }


def test_extract_rows_returns_all_csv_columns():
    doc = _make_sample_concurrency_doc()
    rows = agg_cli.extract_rows(doc)
    assert len(rows) == 2
    for row in rows:
        for col in agg_cli.CSV_COLUMNS:
            assert col in row, f"Missing CSV column: {col}"


def test_extract_rows_concurrency_values():
    doc = _make_sample_concurrency_doc()
    rows = agg_cli.extract_rows(doc)
    assert [r["concurrency"] for r in rows] == [1, 2]


# ── JSON schema (top-level) ───────────────────────────────────────────────────

def test_json_top_level_keys():
    doc = _make_sample_concurrency_doc()
    for key in ["metadata", "configuration", "experiments"]:
        assert key in doc


def test_experiment_entry_has_required_keys():
    doc = _make_sample_concurrency_doc()
    for experiment in doc["experiments"]:
        assert "concurrency" in experiment
        assert "raw_requests" in experiment
        assert "summary" in experiment


# ── extended concurrency sweep (saturation study: 1, 2, 4, 6, 8) ─────────────

def test_validate_concurrency_config_accepts_extended_levels():
    cfg = ConcurrencyBenchmarkConfig(
        server_url="http://x", model_id="m", prompt="p",
        concurrency_levels=[1, 2, 4, 6, 8],
    )
    validate_concurrency_config(cfg)  # should not raise


def test_extract_rows_covers_five_concurrency_levels():
    doc = _make_sample_concurrency_doc([1, 2, 4, 6, 8])
    rows = agg_cli.extract_rows(doc)
    assert [r["concurrency"] for r in rows] == [1, 2, 4, 6, 8]
    for row in rows:
        for col in agg_cli.CSV_COLUMNS:
            assert col in row, f"Missing CSV column: {col}"


def test_aggregate_zero_success_level_does_not_crash():
    # An entire concurrency level with no successful requests (e.g. all timed out).
    results = [_make_result(None, 30000.0, None, success=False, error_type="timeout") for _ in range(5)]
    summary = aggregate_concurrency_results(results, duration_seconds=30.0)
    assert summary["request_count"] == 5
    assert summary["success_count"] == 0
    assert summary["failure_count"] == 5
    assert summary["failure_rate"] == pytest.approx(1.0)
    assert summary["ttft_mean_ms"] is None
    assert summary["e2e_mean_ms"] is None
    assert summary["approx_tpot_mean_ms"] is None
    assert summary["mean_completion_tokens"] is None
    assert summary["request_throughput_rps"] == pytest.approx(0.0)
    assert summary["output_token_throughput_tps"] == pytest.approx(0.0)


# ── relative-scaling (saturation) calculations ───────────────────────────────

def test_ratio_basic():
    assert agg_cli._ratio(20.0, 10.0) == pytest.approx(2.0)


def test_ratio_zero_baseline_returns_blank():
    assert agg_cli._ratio(20.0, 0.0) == ""


def test_ratio_non_numeric_returns_blank():
    assert agg_cli._ratio("", 10.0) == ""
    assert agg_cli._ratio(10.0, "") == ""


def test_add_relative_scaling_baseline_is_one():
    rows = [
        {"concurrency": 1, "output_token_throughput_tps": 70.0, "ttft_median_ms": 30.0, "e2e_median_ms": 300.0},
        {"concurrency": 4, "output_token_throughput_tps": 210.0, "ttft_median_ms": 45.0, "e2e_median_ms": 450.0},
    ]
    result = agg_cli.add_relative_scaling(rows)
    baseline_row = result[0]
    assert baseline_row["throughput_scaling_vs_c1"] == pytest.approx(1.0)
    assert baseline_row["ttft_median_increase_vs_c1"] == pytest.approx(1.0)
    assert baseline_row["e2e_median_increase_vs_c1"] == pytest.approx(1.0)


def test_add_relative_scaling_reflects_saturation():
    # Throughput only doubles (not 4x) while latency triples -> scaling < concurrency ratio.
    rows = [
        {"concurrency": 1, "output_token_throughput_tps": 70.0, "ttft_median_ms": 30.0, "e2e_median_ms": 300.0},
        {"concurrency": 4, "output_token_throughput_tps": 140.0, "ttft_median_ms": 90.0, "e2e_median_ms": 900.0},
    ]
    result = agg_cli.add_relative_scaling(rows)
    saturated_row = result[1]
    assert saturated_row["throughput_scaling_vs_c1"] == pytest.approx(2.0)
    assert saturated_row["ttft_median_increase_vs_c1"] == pytest.approx(3.0)
    assert saturated_row["e2e_median_increase_vs_c1"] == pytest.approx(3.0)
    # Latency growing faster than throughput is the saturation signal readers should look for.
    assert saturated_row["e2e_median_increase_vs_c1"] > saturated_row["throughput_scaling_vs_c1"]


def test_add_relative_scaling_missing_baseline_leaves_blank():
    rows = [
        {"concurrency": 2, "output_token_throughput_tps": 100.0, "ttft_median_ms": 40.0, "e2e_median_ms": 400.0},
    ]
    result = agg_cli.add_relative_scaling(rows)
    assert result[0]["throughput_scaling_vs_c1"] == ""
    assert result[0]["ttft_median_increase_vs_c1"] == ""
    assert result[0]["e2e_median_increase_vs_c1"] == ""


# ── plot generation across the extended sweep ────────────────────────────────

def test_plot_all_covers_five_concurrency_levels(tmp_path):
    import plot_concurrency_results as plot_cli

    csv_path = tmp_path / "phase2_concurrency_summary.csv"
    header = plot_cli.REQUIRED_PLOT_COLS
    rows = []
    for i, c in enumerate([1, 2, 4, 6, 8]):
        rows.append(
            {
                "concurrency": c,
                "ttft_median_ms": 30.0 + i * 2,
                "ttft_p95_ms": 32.0 + i * 2,
                "e2e_median_ms": 300.0 + i * 20,
                "e2e_p95_ms": 320.0 + i * 20,
                "request_throughput_rps": 3.0 + i,
                "output_token_throughput_tps": 70.0 + i * 10,
            }
        )
    import csv as csv_module
    with open(csv_path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "plots"
    paths = plot_cli.plot_all(csv_path, out_dir)
    assert len(paths) == 4
    names = {p.name for p in paths}
    assert names == {
        "phase2_ttft_vs_concurrency.png",
        "phase2_e2e_vs_concurrency.png",
        "phase2_throughput_vs_concurrency.png",
        "phase2_latency_throughput_tradeoff.png",
    }
    for p in paths:
        assert p.exists()
