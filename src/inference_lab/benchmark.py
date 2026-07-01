import statistics
import torch
from .config import InferenceConfig
from .memory import reset_peak_stats, peak_allocated_mb, peak_reserved_mb


def validate_benchmark_config(cfg: InferenceConfig) -> None:
    if cfg.measurement_iterations is None or cfg.measurement_iterations < 1:
        raise ValueError(
            f"measurement_iterations must be >= 1, got {cfg.measurement_iterations!r}"
        )
    if cfg.warmup_iterations is not None and cfg.warmup_iterations < 0:
        raise ValueError(
            f"warmup_iterations must be >= 0, got {cfg.warmup_iterations}"
        )


def _timed_forward(model, **kwargs):
    """
    Run one model forward pass and return (outputs, elapsed_ms).

    CUDA events bracket the forward call.  end.record() is queued immediately
    after the Python call returns (the GPU work is asynchronous), then
    torch.cuda.synchronize() waits for the GPU to finish before we read
    elapsed_time.  Synchronization is therefore AFTER the measured interval,
    not inside it.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    outputs = model(**kwargs)
    end.record()
    torch.cuda.synchronize()
    return outputs, start.elapsed_time(end)


def _run_prefill(model, input_ids, attention_mask):
    """Full-prompt forward pass with KV cache. Returns (first_token [1,1], past_kv, ms)."""
    outputs, latency_ms = _timed_forward(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    first_token = outputs.logits[0, -1, :].argmax().reshape(1, 1)
    return first_token, outputs.past_key_values, latency_ms


def _run_decode(model, first_token, past_key_values, max_steps, eos_token_id):
    """
    Decode one token at a time, reusing the KV cache from prefill.

    Returns:
        token_ids          - all generated token IDs, starting with first_token
        step_latencies_ms  - one timing entry per decode forward pass
    """
    token_ids = [first_token.item()]
    step_latencies_ms = []
    current = first_token       # shape [1, 1]
    past_kv = past_key_values

    for _ in range(max_steps):
        outputs, step_ms = _timed_forward(
            model,
            input_ids=current,
            past_key_values=past_kv,
            use_cache=True,
        )
        step_latencies_ms.append(step_ms)
        next_token = outputs.logits[0, -1, :].argmax().reshape(1, 1)
        past_kv = outputs.past_key_values
        token_ids.append(next_token.item())
        current = next_token
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return token_ids, step_latencies_ms


def run_single_iteration(model, tokenizer, cfg: InferenceConfig) -> dict:
    """Run prefill + decode for one iteration and return all metrics."""
    encoded = tokenizer(cfg.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(cfg.device)
    attn_mask = encoded.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(cfg.device)
    input_token_count = input_ids.shape[1]

    # Reset peak tracking so per-iteration peak reflects only this iteration's
    # execution (model weights are already resident and counted in the baseline).
    reset_peak_stats()

    with torch.inference_mode():
        first_token, past_kv, prefill_ms = _run_prefill(model, input_ids, attn_mask)
        token_ids, step_latencies_ms = _run_decode(
            model,
            first_token,
            past_kv,
            max_steps=max(cfg.max_new_tokens - 1, 0),
            eos_token_id=tokenizer.eos_token_id,
        )

    n_decode = len(step_latencies_ms)
    decode_total_ms = sum(step_latencies_ms)
    e2e_ms = prefill_ms + decode_total_ms

    if n_decode > 0:
        mean_decode_ms = statistics.mean(step_latencies_ms)
        median_decode_ms = statistics.median(step_latencies_ms)
        sorted_steps = sorted(step_latencies_ms)
        p95_idx = min(int(n_decode * 0.95), n_decode - 1)
        p95_decode_ms = sorted_steps[p95_idx]
        decode_tps = n_decode / (decode_total_ms / 1000.0)
    else:
        mean_decode_ms = median_decode_ms = p95_decode_ms = decode_tps = 0.0

    return {
        "input_tokens": input_token_count,
        "generated_tokens": len(token_ids),
        "prefill_latency_ms": round(prefill_ms, 3),
        "decode_total_latency_ms": round(decode_total_ms, 3),
        "decode_token_latencies_ms": [round(x, 3) for x in step_latencies_ms],
        "mean_decode_token_latency_ms": round(mean_decode_ms, 3),
        "median_decode_token_latency_ms": round(median_decode_ms, 3),
        "p95_decode_token_latency_ms": round(p95_decode_ms, 3),
        "decode_tokens_per_second": round(decode_tps, 3),
        "e2e_latency_ms": round(e2e_ms, 3),
        "peak_cuda_allocated_mb": round(peak_allocated_mb(), 1),
        "peak_cuda_reserved_mb": round(peak_reserved_mb(), 1),
    }


def run_benchmark(model, tokenizer, cfg: InferenceConfig) -> list[dict]:
    """
    Run warmup iterations (results discarded) then measurement iterations.
    Returns a list of per-iteration metric dicts.
    """
    validate_benchmark_config(cfg)

    warmup_n = cfg.warmup_iterations or 0
    for _ in range(warmup_n):
        run_single_iteration(model, tokenizer, cfg)

    results = []
    for i in range(cfg.measurement_iterations):
        result = run_single_iteration(model, tokenizer, cfg)
        result["iteration"] = i
        results.append(result)

    return results
