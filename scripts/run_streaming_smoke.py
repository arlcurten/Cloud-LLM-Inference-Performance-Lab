"""Phase 2A streaming online inference smoke test.

Sends a single streaming request to a vLLM OpenAI-compatible server and
prints text as it arrives, measuring provisional time-to-first-token (TTFT)
and end-to-end latency.

Limitation: the OpenAI-compatible streaming protocol exposes text chunks,
not individual model tokens. A chunk may contain zero, one, or more tokens
depending on the server's tokenizer and detokenization behavior. TTFT here
is measured as time-to-first-chunk, and inter-chunk latency is reported
instead of exact inter-token latency (TPOT). Do not treat these as exact
per-token measurements.
"""
import argparse
import json
import sys
import time

import requests

from run_online_smoke import build_url, check_server_reachable


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
    content delta in this chunk (e.g. a role-only chat delta).
    """
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return choice.get("text") or ""
    delta = choice.get("delta") or {}
    return delta.get("content") or ""


def compute_streaming_metrics(request_start: float, chunk_times: list) -> dict:
    """Compute provisional timing metrics from chunk arrival timestamps.

    request_start and chunk_times must be from the same monotonic clock
    (e.g. time.perf_counter()). chunk_times must contain only the arrival
    times of chunks with non-empty text content.
    """
    if not chunk_times:
        raise ValueError("No content chunks received")
    ttft_ms = (chunk_times[0] - request_start) * 1000
    e2e_latency_ms = (chunk_times[-1] - request_start) * 1000
    inter_chunk_latencies_ms = [
        (chunk_times[i] - chunk_times[i - 1]) * 1000 for i in range(1, len(chunk_times))
    ]
    return {
        "ttft_ms": ttft_ms,
        "e2e_latency_ms": e2e_latency_ms,
        "num_chunks": len(chunk_times),
        "inter_chunk_latencies_ms": inter_chunk_latencies_ms,
    }


def run_streaming_completion(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float = 60.0):
    """Stream a completion request. Returns (full_text, chunk_times, request_start)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    chunk_times = []
    text_parts = []
    request_start = time.perf_counter()
    with requests.post(
        build_url(base_url, "/v1/completions"), json=payload, timeout=timeout, stream=True
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            chunk = parse_sse_line(raw_line)
            if chunk is None:
                continue
            delta = extract_text_delta(chunk)
            if delta:
                chunk_times.append(time.perf_counter())
                text_parts.append(delta)
                print(delta, end="", flush=True)
    print()
    return "".join(text_parts), chunk_times, request_start


def main():
    parser = argparse.ArgumentParser(description="Streaming online inference smoke test")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--model", required=True, help="Model id as registered by the server")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    print(f"Checking server at {args.url} ...")
    try:
        check_server_reachable(args.url)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: could not connect to {args.url}. Is the server running?", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: server reachability check failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("Server reachable. Streaming response:\n")
    try:
        _, chunk_times, request_start = run_streaming_completion(
            args.url, args.model, args.prompt, args.max_tokens
        )
    except requests.exceptions.RequestException as e:
        print(f"ERROR: request failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        metrics = compute_streaming_metrics(request_start, chunk_times)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Streaming Smoke Result (provisional) ---")
    print(f"Provisional TTFT (time to first chunk): {metrics['ttft_ms']:.1f} ms")
    print(f"End-to-end latency: {metrics['e2e_latency_ms']:.1f} ms")
    print(f"Number of chunks: {metrics['num_chunks']}")
    if metrics["inter_chunk_latencies_ms"]:
        avg_inter = sum(metrics["inter_chunk_latencies_ms"]) / len(metrics["inter_chunk_latencies_ms"])
        print(f"Mean inter-chunk latency: {avg_inter:.1f} ms (not exact TPOT — see module docstring)")


if __name__ == "__main__":
    main()
