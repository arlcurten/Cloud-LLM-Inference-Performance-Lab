# Cloud LLM Inference Performance Lab

A personal study project for learning and benchmarking local LLM inference on consumer GPU hardware.

**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)  
**Primary model:** `google/gemma-3-1b-it`

---

## Phase 1 Scope

- Load Gemma 3 1B Instruct locally via Hugging Face Transformers
- Run a single deterministic generation (smoke test)
- Verify GPU memory usage fits within 4 GB VRAM
- Measure prefill and decode latency manually with CUDA event timing
- Collect repeatable per-iteration metrics and aggregate statistics

---

## Environment Setup

Python 3.12+ and a CUDA-capable GPU are required.

```bash
# Clone the repo
git clone https://github.com/arlcurten/Cloud-LLM-Inference-Performance-Lab.git
cd Cloud-LLM-Inference-Performance-Lab

# Install dependencies (PyTorch must already match your CUDA version)
pip install -r requirements.txt
```

> **PyTorch:** Install the CUDA-enabled wheel from https://pytorch.org before running the above if PyTorch is not already installed.

---

## Hugging Face Login

Gemma 3 is a gated model. You must accept the license on Hugging Face and log in:

```bash
huggingface-cli login
```

---

## Run the Smoke Test

```bash
python scripts/run_smoke_inference.py --config configs/phase1_smoke.yaml
```

### Expected Output

```
Loading model: google/gemma-3-1b-it

--- Smoke Inference Result ---
GPU              : NVIDIA GeForce RTX 3050 Laptop GPU
Input tokens     : 8
Generated tokens : 22
Generated text   :
The capital of France is Paris.

Final Answer: The final answer is $\boxed{Paris}$
CUDA allocated   : 1915.3 MB
CUDA reserved    : 1946.0 MB
```

*(Exact token counts and memory figures may vary slightly.)*

---

## Run the Benchmark

```bash
python scripts/run_benchmark.py --config configs/phase1_baseline.yaml
```

Results are written to `results/raw/benchmark_<timestamp>.json`.

### What the benchmark measures

**Prefill** — one full forward pass over the entire prompt with `use_cache=True`.  
The next token is selected by greedy argmax of the last-position logits.

**Decode** — one forward pass per token, feeding a single token and the KV cache
from the previous step. Each step is timed independently.

**Timing method** — CUDA events (`torch.cuda.Event`). `start.record()` and
`end.record()` bracket each forward pass; `torch.cuda.synchronize()` is called
after `end.record()` to wait for GPU completion before reading `elapsed_time`.
This isolates GPU execution time and avoids including CPU-to-GPU synchronization
overhead inside the measured window.

**Warm-up** — the first `warmup_iterations` runs are discarded. They cover
first-run CUDA kernel compilation and any OS/driver cold-start effects.
Model loading and tokenizer loading are never included in latency.

**Metrics per iteration:** input token count, generated token count, prefill
latency (ms), decode total latency (ms), per-step mean/median/P95 (ms), decode
tokens/s, end-to-end latency (ms), peak CUDA allocated and reserved (MB).

**Summary statistics** across all measured iterations: mean, median, P95, min, max.

### Known limitations (small local GPU)

- With only 10 measured iterations, P95 equals the maximum (no interpolation).
- Laptop GPU thermals cause high variance; later iterations can be 1.5× slower
  than the first due to thermal throttling.
- Decode throughput (~18–20 tok/s at float16) is memory-bandwidth-limited on a
  4 GB VRAM GPU; it does not reflect server-grade hardware.
- `batch_size` is fixed at 1; batched throughput is not measured yet.

---

## Run Unit Tests

```bash
pytest src_test/
```

---

## Model Storage

Models are downloaded to `models/<org>/<model-name>/` (a flat, easy-to-browse layout).  
This folder is git-ignored. To free disk space, simply delete the subfolder:

```bash
rm -rf models/google/gemma-3-1b-it
```

To use the default HuggingFace cache instead, remove the `model_dir` line from the YAML config.

---

## Project Structure

```
configs/                   YAML experiment configurations
models/                    Local model downloads (git-ignored)
src/inference_lab/
  config.py                Config dataclass and YAML loader
  model_loader.py          Tokenizer and model loading
  benchmark.py             Prefill/decode timing logic
  metrics.py               Stats aggregation (mean/median/P95/min/max)
  memory.py                CUDA memory helpers
  system_info.py           Environment metadata collection
scripts/
  run_smoke_inference.py   Quick single-run sanity check
  run_benchmark.py         Full timed benchmark with JSON output
src_test/                  Unit tests (no model required)
results/raw/               Timestamped JSON benchmark outputs (git-ignored)
results/plots/             Generated plots (git-ignored)
```
