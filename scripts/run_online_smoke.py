"""Phase 2A non-streaming online inference smoke test.

Sends a single non-streaming request to a vLLM OpenAI-compatible server and
reports HTTP status, generated text, request latency, and token counts.
"""
import argparse
import sys
import time

import requests


def build_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def check_server_reachable(base_url: str, timeout: float = 5.0) -> dict:
    resp = requests.get(build_url(base_url, "/v1/models"), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def extract_completion_text(body: dict) -> str:
    choices = body.get("choices")
    if not choices:
        raise ValueError(f"Response missing 'choices': {body}")
    text = choices[0].get("text")
    if text is None:
        raise ValueError(f"Response choice missing 'text': {choices[0]}")
    return text


def run_completion(
    base_url: str, model: str, prompt: str, max_tokens: int, timeout: float = 60.0
):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    resp = requests.post(build_url(base_url, "/v1/completions"), json=payload, timeout=timeout)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return resp, elapsed_ms


def main():
    parser = argparse.ArgumentParser(description="Non-streaming online inference smoke test")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--model", required=True, help="Model id as registered by the server")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    print(f"Checking server at {args.url} ...")
    try:
        models = check_server_reachable(args.url)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: could not connect to {args.url}. Is the server running?", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: server reachability check failed: {e}", file=sys.stderr)
        sys.exit(1)

    available = [m["id"] for m in models.get("data", [])]
    print(f"Server reachable. Models: {available}")

    try:
        resp, elapsed_ms = run_completion(args.url, args.model, args.prompt, args.max_tokens)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: request failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"HTTP status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"ERROR response body: {resp.text}", file=sys.stderr)
        sys.exit(1)

    try:
        body = resp.json()
        text = extract_completion_text(body)
    except (ValueError, requests.exceptions.JSONDecodeError) as e:
        print(f"ERROR: could not parse response: {e}", file=sys.stderr)
        sys.exit(1)

    usage = body.get("usage", {})

    print("\n--- Non-Streaming Smoke Result ---")
    print(f"Generated text:\n{text}")
    print(f"\nTotal request latency: {elapsed_ms:.1f} ms")
    if "prompt_tokens" in usage:
        print(f"Prompt tokens: {usage['prompt_tokens']}")
    if "completion_tokens" in usage:
        print(f"Completion tokens: {usage['completion_tokens']}")


if __name__ == "__main__":
    main()
