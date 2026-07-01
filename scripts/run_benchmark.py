"""Run a prefill/decode benchmark and write a timestamped JSON result file."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference_lab.benchmark import run_benchmark, validate_benchmark_config
from inference_lab.config import load_config
from inference_lab.metrics import aggregate_iterations
from inference_lab.model_loader import load_model_and_tokenizer
from inference_lab.system_info import collect as collect_system_info


def _print_summary(summary: dict, n_iters: int) -> None:
    def row(label, key, unit="ms"):
        s = summary[key]
        print(
            f"  {label:<32} mean={s['mean']:>8.2f}  median={s['median']:>8.2f}"
            f"  p95={s['p95']:>8.2f}  min={s['min']:>8.2f}  max={s['max']:>8.2f}  [{unit}]"
        )

    print(f"\n{'='*74}")
    print(f"  Benchmark Summary  ({n_iters} measured iterations)")
    print(f"{'='*74}")
    print(f"  Input tokens           : {summary['input_tokens']}")
    print(f"  Generated tokens (mean): {summary['generated_tokens']['mean']:.1f}")
    print()
    row("Prefill latency", "prefill_latency_ms")
    row("Decode total latency", "decode_total_latency_ms")
    row("Mean decode step latency", "mean_decode_token_latency_ms")
    row("Median decode step latency", "median_decode_token_latency_ms")
    row("P95 decode step latency", "p95_decode_token_latency_ms")
    row("Decode tokens/s", "decode_tokens_per_second", unit="tok/s")
    row("E2E latency", "e2e_latency_ms")
    print()
    row("Peak CUDA allocated", "peak_cuda_allocated_mb", unit="MB")
    row("Peak CUDA reserved", "peak_cuda_reserved_mb", unit="MB")
    print(f"{'='*74}\n")


def main():
    parser = argparse.ArgumentParser(description="Gemma prefill/decode benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. A CUDA-capable GPU is required.", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    try:
        validate_benchmark_config(cfg)
    except ValueError as e:
        print(f"ERROR: Invalid benchmark config — {e}", file=sys.stderr)
        sys.exit(1)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    experiment_name = Path(args.config).stem
    print(f"Experiment    : {experiment_name}")
    print(f"Loading model : {cfg.model_id}")
    tokenizer, model = load_model_and_tokenizer(cfg)

    print(f"Running {cfg.warmup_iterations} warm-up iteration(s)...")
    print(f"Running {cfg.measurement_iterations} measured iteration(s)...")

    iterations = run_benchmark(model, tokenizer, cfg)
    summary = aggregate_iterations(iterations)

    _print_summary(summary, len(iterations))

    sysinfo = collect_system_info()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_doc = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_name": experiment_name,
            "git_commit": sysinfo["git_commit"],
            "model_id": cfg.model_id,
            "dtype": cfg.dtype,
            "device": cfg.device,
            "gpu_name": sysinfo["gpu_name"],
            "total_vram_mb": sysinfo["total_vram_mb"],
            "python_version": sysinfo["python_version"],
            "pytorch_version": sysinfo["pytorch_version"],
            "transformers_version": sysinfo["transformers_version"],
            "cuda_version": sysinfo["cuda_version"],
        },
        "configuration": {
            "prompt": cfg.prompt,
            "max_new_tokens": cfg.max_new_tokens,
            "warmup_iterations": cfg.warmup_iterations,
            "measurement_iterations": cfg.measurement_iterations,
            "do_sample": cfg.do_sample,
            "seed": cfg.seed,
        },
        "iterations": iterations,
        "summary": summary,
    }

    out_dir = Path("results/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"benchmark_{experiment_name}_{timestamp}.json"
    out_path.write_text(json.dumps(result_doc, indent=2))
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
