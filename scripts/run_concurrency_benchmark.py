"""Phase 2B: async closed-loop concurrency benchmark.

Runs each configured concurrency level (warm-up, then measured requests)
against a live vLLM OpenAI-compatible server and writes a timestamped JSON
result file under results/raw/. See src/inference_lab/online_benchmark.py
for the closed-loop scheduling and metric definitions (TTFT, E2E,
approx_tpot_ms, throughput).
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference_lab.online_benchmark import (
    aggregate_concurrency_results,
    load_concurrency_config,
    run_concurrency_level,
    validate_concurrency_config,
)
from inference_lab.server_config import load_server_config
from inference_lab.server_info import collect_server_metadata


def _check_server(base_url: str) -> None:
    resp = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=5.0)
    resp.raise_for_status()


def _print_level_summary(concurrency: int, summary: dict) -> None:
    print(f"\n{'=' * 74}")
    print(f"  Concurrency = {concurrency}")
    print(f"{'=' * 74}")
    print(
        f"  Requests       : {summary['request_count']}"
        f"  (success={summary['success_count']}  failure={summary['failure_count']}"
        f"  rate={summary['failure_rate']:.1%})"
    )
    print(
        f"  TTFT   (ms)    : mean={summary['ttft_mean_ms']}  median={summary['ttft_median_ms']}"
        f"  p95={summary['ttft_p95_ms']}  p99={summary['ttft_p99_ms']}"
    )
    print(
        f"  E2E    (ms)    : mean={summary['e2e_mean_ms']}  median={summary['e2e_median_ms']}"
        f"  p95={summary['e2e_p95_ms']}  p99={summary['e2e_p99_ms']}"
    )
    print(
        f"  approx TPOT(ms): mean={summary['approx_tpot_mean_ms']}"
        f"  median={summary['approx_tpot_median_ms']}  p95={summary['approx_tpot_p95_ms']}"
    )
    print(
        f"  Throughput     : {summary['request_throughput_rps']} req/s"
        f"   {summary['output_token_throughput_tps']} tok/s"
    )
    print(f"  Duration       : {summary['benchmark_duration_seconds']} s")


async def _run_all(cfg) -> list:
    experiments = []
    for concurrency in cfg.concurrency_levels:
        if cfg.warmup_requests > 0:
            print(f"\n--- Concurrency {concurrency}: warm-up ({cfg.warmup_requests} requests) ---")
            warmup_concurrency = min(concurrency, cfg.warmup_requests)
            await run_concurrency_level(
                cfg.server_url,
                cfg.model_id,
                cfg.prompt,
                cfg.max_tokens,
                cfg.temperature,
                cfg.request_timeout_seconds,
                warmup_concurrency,
                cfg.warmup_requests,
            )

        print(f"--- Concurrency {concurrency}: measuring ({cfg.requests_per_level} requests) ---")
        level_start = time.perf_counter()
        results = await run_concurrency_level(
            cfg.server_url,
            cfg.model_id,
            cfg.prompt,
            cfg.max_tokens,
            cfg.temperature,
            cfg.request_timeout_seconds,
            concurrency,
            cfg.requests_per_level,
        )
        level_duration = time.perf_counter() - level_start

        summary = aggregate_concurrency_results(results, level_duration)
        _print_level_summary(concurrency, summary)

        experiments.append(
            {
                "concurrency": concurrency,
                "raw_requests": [r.to_dict() for r in results],
                "summary": summary,
            }
        )
    return experiments


def main():
    parser = argparse.ArgumentParser(description="Phase 2B async concurrency benchmark")
    parser.add_argument("--config", default="configs/phase2_concurrency.yaml")
    parser.add_argument(
        "--server-config",
        default="configs/phase2_vllm_local.yaml",
        help="Server config (used only to report max_num_seqs etc. in result metadata)",
    )
    args = parser.parse_args()

    cfg = load_concurrency_config(args.config)
    try:
        validate_concurrency_config(cfg)
    except ValueError as e:
        print(f"ERROR: Invalid concurrency config — {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Checking server at {cfg.server_url} ...")
    try:
        _check_server(cfg.server_url)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: could not reach server at {cfg.server_url}: {e}", file=sys.stderr)
        sys.exit(1)
    print("Server reachable.")

    server_cfg = load_server_config(args.server_config)
    print(
        f"Server config max_num_seqs = {server_cfg.max_num_seqs}. Confirm concurrency levels "
        f"{cfg.concurrency_levels} are consistent with this before interpreting results at "
        f"higher concurrency."
    )

    experiments = asyncio.run(_run_all(cfg))

    metadata = collect_server_metadata(server_cfg)
    result_doc = {
        "metadata": metadata,
        "configuration": {
            "prompt": cfg.prompt,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "stream": cfg.stream,
            "concurrency_levels": cfg.concurrency_levels,
            "requests_per_level": cfg.requests_per_level,
            "warmup_requests": cfg.warmup_requests,
            "request_timeout_seconds": cfg.request_timeout_seconds,
        },
        "experiments": experiments,
    }

    out_dir = Path("results/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"concurrency_benchmark_{timestamp}.json"
    out_path.write_text(json.dumps(result_doc, indent=2))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
