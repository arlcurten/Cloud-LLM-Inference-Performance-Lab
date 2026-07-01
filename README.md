# Cloud LLM Inference Performance Lab

A personal study project for learning and benchmarking local LLM inference on consumer GPU hardware.

**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)  
**Primary model:** `google/gemma-3-1b-it`

---

## Phase 1 Scope

- Load Gemma 3 1B Instruct locally via Hugging Face Transformers
- Run a single deterministic generation (smoke test)
- Verify GPU memory usage fits within 4 GB VRAM

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
configs/              YAML experiment configurations
models/               Local model downloads (git-ignored)
src/inference_lab/    Core library (config loader, model loader)
scripts/              Runnable entry points
src_test/             Unit tests
results/raw/          Raw inference outputs (git-ignored)
results/plots/        Generated plots (git-ignored)
```
