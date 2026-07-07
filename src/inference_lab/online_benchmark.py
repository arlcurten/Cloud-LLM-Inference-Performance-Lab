"""Reusable online (HTTP) benchmark logic for Phase 2B concurrency testing.

This module implements a closed-loop concurrent load generator against an
OpenAI-compatible streaming completions endpoint (vLLM). "Closed-loop" means
a fixed number of worker tasks (the concurrency level) each send one request
at a time and immediately send their next request as soon as the previous
one completes — concurrency never exceeds the configured level, and there is
no independent arrival-rate control (that is an open-loop model, out of
scope here).

Metric definitions:

- ttft_ms: time from immediately before sending the HTTP request until the
  first non-empty generated text chunk is received. Metadata-only or empty
  chunks are not treated as the first token.
- e2e_latency_ms: time from immediately before sending the request until the
  stream fully completes.
- approx_tpot_ms: an *approximate* client-side time-per-output-token,
  computed only when completion_tokens > 1, as
  (e2e_latency_ms - ttft_ms) / (completion_tokens - 1). This is not exact
  server-side per-token latency: the OpenAI streaming protocol exposes text
  chunks, not individual model tokens, and a chunk may contain zero, one, or
  more tokens.
- output_tokens_per_second: per-request output throughput
  (completion_tokens / e2e_latency_seconds), not aggregate server
  throughput.
"""
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

import httpx
import yaml


# ── Phase 2B benchmark configuration ─────────────────────────────────────────

@dataclass
class ConcurrencyBenchmarkConfig:
    server_url: str
    model_id: str
    prompt: str
    max_tokens: int = 32
    temperature: float = 0.0
    stream: bool = True
    concurrency_levels: list = field(default_factory=lambda: [1, 2, 4])
    requests_per_level: int = 30
    request_timeout_seconds: float = 30.0
    warmup_requests: int = 3
    seed: int = 42


def load_concurrency_config(path: str | Path) -> ConcurrencyBenchmarkConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ConcurrencyBenchmarkConfig(**data)


def validate_concurrency_config(cfg: ConcurrencyBenchmarkConfig) -> None:
    if not cfg.stream:
        raise ValueError(
            "Phase 2B concurrency benchmark only supports stream=true "
            "(TTFT requires a streaming response)"
        )
    if not cfg.concurrency_levels:
        raise ValueError("concurrency_levels must be non-empty")
    if any(c <= 0 for c in cfg.concurrency_levels):
        raise ValueError(f"concurrency_levels must all be positive, got {cfg.concurrency_levels}")
    if cfg.requests_per_level <= 0:
        raise ValueError(f"requests_per_level must be positive, got {cfg.requests_per_level}")
    if cfg.warmup_requests < 0:
        raise ValueError(f"warmup_requests must be non-negative, got {cfg.warmup_requests}")
    if cfg.request_timeout_seconds <= 0:
        raise ValueError(
            f"request_timeout_seconds must be positive, got {cfg.request_timeout_seconds}"
        )
    if cfg.max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {cfg.max_tokens}")


def build_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def parse_sse_line(line: str):
    """Parse one SSE line from an OpenAI-compatible streaming response.

    Returns the decoded JSON payload dict, or None for blank lines,
    non-data lines, or the terminal "[DONE]" marker.
    """
    if not line:
        return None
    if not line.startswith("data:"):
        return None
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        return None
    return json.loads(data)


def extract_text_delta(chunk: dict) -> str:
    """Extract the text delta from a streaming chunk.

    Supports both /v1/completions ("text") and /v1/chat/completions
    ("delta.content") response shapes. Returns "" if there is no
    content delta in this chunk (e.g. a role-only chat delta, or a
    final usage-only chunk with an empty choices list).
    """
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return choice.get("text") or ""
    delta = choice.get("delta") or {}
    return delta.get("content") or ""


def extract_usage(chunk: dict) -> dict:
    """Extract token usage from a streaming chunk, if present.

    Usage is typically only populated on the final chunk of a stream
    (requires stream_options.include_usage on the request).
    """
    return chunk.get("usage") or {}


# ── per-request metric calculations ──────────────────────────────────────────

def compute_ttft_ms(request_start_time: float, first_meaningful_chunk_time: float) -> float:
    return (first_meaningful_chunk_time - request_start_time) * 1000


def compute_e2e_latency_ms(request_start_time: float, request_end_time: float) -> float:
    return (request_end_time - request_start_time) * 1000


def compute_approx_tpot_ms(
    e2e_latency_ms: float, ttft_ms: Optional[float], completion_tokens: Optional[int]
) -> Optional[float]:
    """Approximate client-side time-per-output-token, excluding the first token.

    Only defined when completion_tokens is known and > 1. See module
    docstring for why this is approximate, not exact per-token latency.
    """
    if completion_tokens is None or completion_tokens <= 1:
        return None
    if ttft_ms is None:
        return None
    return (e2e_latency_ms - ttft_ms) / (completion_tokens - 1)


def compute_output_tokens_per_second(
    completion_tokens: Optional[int], e2e_latency_ms: float
) -> Optional[float]:
    """Per-request output throughput. Not aggregate server throughput."""
    if completion_tokens is None or completion_tokens <= 0 or e2e_latency_ms <= 0:
        return None
    return completion_tokens / (e2e_latency_ms / 1000.0)


# ── request result schema ─────────────────────────────────────────────────────

@dataclass
class RequestResult:
    request_start_time: float
    first_meaningful_chunk_time: Optional[float]
    request_end_time: float
    ttft_ms: Optional[float]
    e2e_latency_ms: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    success: bool
    http_status: Optional[int]
    error_type: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


async def run_single_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> RequestResult:
    """Send one streaming /v1/completions request and time it."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request_start_time = time.perf_counter()
    first_meaningful_chunk_time = None
    prompt_tokens = None
    completion_tokens = None
    http_status = None
    error_type = None
    success = False

    try:
        async with client.stream(
            "POST", build_url(base_url, "/v1/completions"), json=payload, timeout=timeout
        ) as resp:
            http_status = resp.status_code
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                chunk = parse_sse_line(raw_line)
                if chunk is None:
                    continue
                delta = extract_text_delta(chunk)
                if delta and first_meaningful_chunk_time is None:
                    first_meaningful_chunk_time = time.perf_counter()
                usage = extract_usage(chunk)
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
        success = True
    except httpx.TimeoutException:
        error_type = "timeout"
    except httpx.HTTPStatusError as e:
        error_type = f"http_{e.response.status_code}"
    except Exception as e:
        error_type = type(e).__name__
    finally:
        request_end_time = time.perf_counter()

    if success and first_meaningful_chunk_time is None:
        # Stream completed but no content chunk ever arrived: treat as failure.
        success = False
        error_type = error_type or "no_content_received"

    ttft_ms = (
        compute_ttft_ms(request_start_time, first_meaningful_chunk_time)
        if first_meaningful_chunk_time is not None
        else None
    )
    e2e_latency_ms = compute_e2e_latency_ms(request_start_time, request_end_time)

    return RequestResult(
        request_start_time=request_start_time,
        first_meaningful_chunk_time=first_meaningful_chunk_time,
        request_end_time=request_end_time,
        ttft_ms=ttft_ms,
        e2e_latency_ms=e2e_latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        success=success,
        http_status=http_status,
        error_type=error_type,
    )


# ── closed-loop concurrency scheduling ───────────────────────────────────────

async def _worker(
    request_fn: Callable[[], Awaitable[RequestResult]],
    remaining: list,
    lock: asyncio.Lock,
    results: list,
) -> None:
    while True:
        async with lock:
            if remaining[0] <= 0:
                return
            remaining[0] -= 1
        result = await request_fn()
        results.append(result)


async def run_closed_loop(
    request_fn: Callable[[], Awaitable[RequestResult]], concurrency: int, num_requests: int
) -> list:
    """Run num_requests calls to request_fn using exactly `concurrency` workers.

    Each worker sends one request, waits for it to finish, then immediately
    sends its next request. In-flight requests never exceed `concurrency`.
    This is a closed-loop load model, not open-loop arrival-rate control.
    """
    if concurrency <= 0:
        raise ValueError(f"concurrency must be positive, got {concurrency}")
    if num_requests <= 0:
        raise ValueError(f"num_requests must be positive, got {num_requests}")

    results: list = []
    remaining = [num_requests]
    lock = asyncio.Lock()
    workers = [
        asyncio.create_task(_worker(request_fn, remaining, lock, results))
        for _ in range(concurrency)
    ]
    await asyncio.gather(*workers)
    return results


async def run_concurrency_level(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    concurrency: int,
    num_requests: int,
) -> list:
    """Closed-loop benchmark of a single concurrency level against a live server."""
    async with httpx.AsyncClient() as client:
        async def request_fn():
            return await run_single_request(
                client, base_url, model, prompt, max_tokens, temperature, timeout
            )

        return await run_closed_loop(request_fn, concurrency, num_requests)


# ── aggregation ───────────────────────────────────────────────────────────────

def _percentile(values: list, pct: float) -> float:
    """Nearest-rank percentile, clamped to the last index (consistent with
    inference_lab.metrics.compute_stats). pct is in [0, 100]."""
    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    s = sorted(values)
    idx = min(int(len(s) * pct / 100), len(s) - 1)
    return s[idx]


def aggregate_concurrency_results(results: list, duration_seconds: float) -> dict:
    """Aggregate a list of RequestResult (or equivalent dicts) for one concurrency level.

    duration_seconds is the benchmark wall-clock duration for this
    concurrency level (from just before the first request was issued to
    just after the last one completed), used for throughput calculations.
    """
    if not results:
        raise ValueError("cannot aggregate an empty list of results")

    def _get(r, key):
        return r[key] if isinstance(r, dict) else getattr(r, key)

    request_count = len(results)
    successes = [r for r in results if _get(r, "success")]
    success_count = len(successes)
    failure_count = request_count - success_count
    failure_rate = failure_count / request_count

    ttft_values = [_get(r, "ttft_ms") for r in successes if _get(r, "ttft_ms") is not None]
    e2e_values = [_get(r, "e2e_latency_ms") for r in successes]
    completion_tokens_values = [
        _get(r, "completion_tokens") for r in successes if _get(r, "completion_tokens") is not None
    ]
    tpot_values = [
        compute_approx_tpot_ms(
            _get(r, "e2e_latency_ms"), _get(r, "ttft_ms"), _get(r, "completion_tokens")
        )
        for r in successes
    ]
    tpot_values = [v for v in tpot_values if v is not None]

    total_completion_tokens = sum(completion_tokens_values) if completion_tokens_values else 0

    summary = {
        "request_count": request_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "failure_rate": round(failure_rate, 4),
        "benchmark_duration_seconds": round(duration_seconds, 4),
        "request_throughput_rps": round(success_count / duration_seconds, 4) if duration_seconds > 0 else None,
        "output_token_throughput_tps": (
            round(total_completion_tokens / duration_seconds, 4) if duration_seconds > 0 else None
        ),
        "total_completion_tokens": total_completion_tokens,
        "mean_completion_tokens": (
            round(statistics.mean(completion_tokens_values), 3) if completion_tokens_values else None
        ),
        "ttft_mean_ms": round(statistics.mean(ttft_values), 3) if ttft_values else None,
        "ttft_median_ms": round(statistics.median(ttft_values), 3) if ttft_values else None,
        "ttft_p95_ms": round(_percentile(ttft_values, 95), 3) if ttft_values else None,
        "ttft_p99_ms": round(_percentile(ttft_values, 99), 3) if ttft_values else None,
        "e2e_mean_ms": round(statistics.mean(e2e_values), 3) if e2e_values else None,
        "e2e_median_ms": round(statistics.median(e2e_values), 3) if e2e_values else None,
        "e2e_p95_ms": round(_percentile(e2e_values, 95), 3) if e2e_values else None,
        "e2e_p99_ms": round(_percentile(e2e_values, 99), 3) if e2e_values else None,
        "approx_tpot_mean_ms": round(statistics.mean(tpot_values), 3) if tpot_values else None,
        "approx_tpot_median_ms": round(statistics.median(tpot_values), 3) if tpot_values else None,
        "approx_tpot_p95_ms": round(_percentile(tpot_values, 95), 3) if tpot_values else None,
    }
    return summary
