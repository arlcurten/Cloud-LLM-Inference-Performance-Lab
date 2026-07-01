# Cloud LLM Inference Performance Lab

A personal study project for learning and benchmarking local LLM inference on consumer GPU hardware.

**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)  
**Primary model:** `google/gemma-3-1b-it`

> **Note:** Results from this laptop GPU validate the benchmark workflow and methodology.
> They do not represent production cloud-inference capacity.

---

## Phase 1 Scope

- Load Gemma 3 1B Instruct locally via Hugging Face Transformers
- Run a single deterministic generation (smoke test)
- Verify GPU memory usage fits within 4 GB VRAM
- Measure prefill and decode latency manually with CUDA event timing
- Collect repeatable per-iteration metrics and aggregate statistics across three workloads

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

Results are written to `results/raw/benchmark_<experiment>_<timestamp>.json`.

### Metric definitions

**Prefill latency** — time (ms) for one full forward pass over the entire prompt
with `use_cache=True`. Includes KV cache construction for the input sequence.

**Decode token latency** — time (ms) per individual token generation step during
the decode phase. Each step feeds one token and the previous KV cache.

**Decode tokens/s** — number of decode steps divided by total decode latency.
This measures decode-phase throughput and excludes prefill time.

**E2E latency** — sum of prefill latency and all decode step latencies. Excludes
model loading, tokenization, and CPU overhead between steps.

### Timing method

CUDA events (`torch.cuda.Event`) bracket each forward pass. `end.record()` is
queued immediately after the model call returns; `torch.cuda.synchronize()` then
waits for GPU completion before `elapsed_time` is read. Synchronization is
outside the measured interval — the measured time is GPU execution time only.

### Warm-up

The first `warmup_iterations` runs are discarded. They absorb first-run CUDA
kernel compilation and driver initialization. Model and tokenizer loading are
never included in any latency measurement.

---

## Phase 1 Baseline Workloads

Three workloads cover short, medium, and long input lengths at batch size 1.

| Config | Input tokens | Requested output tokens |
|---|---|---|
| `phase1_short.yaml` | 30 | 32 |
| `phase1_medium.yaml` | 117 | 64 |
| `phase1_long.yaml` | 523 | 100 |

Each run uses 3 warm-up iterations and 10 measured iterations with deterministic
(greedy) decoding.

```bash
python scripts/run_benchmark.py --config configs/phase1_short.yaml
python scripts/run_benchmark.py --config configs/phase1_medium.yaml
python scripts/run_benchmark.py --config configs/phase1_long.yaml
```

---

## Aggregate Results to CSV

```bash
python scripts/aggregate_results.py results/raw/
```

Or pass specific files:

```bash
python scripts/aggregate_results.py \
  results/raw/benchmark_phase1_short_*.json \
  results/raw/benchmark_phase1_medium_*.json \
  results/raw/benchmark_phase1_long_*.json
```

Writes a CSV to `results/processed/summary_<timestamp>.csv`.

---

## Generate Plots

```bash
python scripts/plot_results.py results/processed/summary_<timestamp>.csv
```

Produces three PNG files under `results/plots/`:

- `prefill_latency_vs_input_tokens.png`
- `decode_tokens_per_second_vs_workload.png`
- `e2e_latency_vs_workload.png`

---

## Run Unit Tests

```bash
pytest src_test/
```

---

## Known Limitations (RTX 3050 Laptop GPU)

- **Sample size:** With only 10 measured iterations, the P95 estimate is
  unstable and should be interpreted cautiously. It will equal the maximum
  observation when n=10.
- **Latency variance:** Observed run-to-run variance may be caused by laptop GPU
  thermal behavior, power management, background system activity, or normal
  runtime variance. Temperature, clock, and power data were not collected.
- **Throughput:** Decode throughput (~8–18 tok/s at float16 across workloads) is
  memory-bandwidth-limited on a 4 GB VRAM laptop GPU and does not reflect
  server-grade hardware.
- **Batch size:** Fixed at 1; batched throughput is not measured.
- **Scope:** These results validate the benchmark workflow. They are not
  representative of production cloud inference capacity.

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
  aggregate_results.py     Merge JSON files into a summary CSV
  plot_results.py          Generate PNG plots from CSV
src_test/                  Unit tests (no model required)
results/raw/               Timestamped JSON benchmark outputs (git-ignored)
results/processed/         Aggregated CSV files (git-ignored)
results/plots/             Generated PNG plots (git-ignored)
```
