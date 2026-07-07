#!/usr/bin/env bash
# Starts the vLLM OpenAI-compatible API server for Phase 2A local serving.
# Reads configs/phase2_vllm_local.yaml (or a path passed as $1) and runs
# vLLM from the isolated .venv-vllm virtual environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="${1:-$REPO_ROOT/configs/phase2_vllm_local.yaml}"
VENV_PYTHON="$REPO_ROOT/.venv-vllm/bin/python"
VLLM_BIN="$REPO_ROOT/.venv-vllm/bin/vllm"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "ERROR: vLLM venv not found at $REPO_ROOT/.venv-vllm" >&2
    echo "See README Phase 2 setup: python3 -m venv .venv-vllm && .venv-vllm/bin/pip install vllm" >&2
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: config file not found: $CONFIG_PATH" >&2
    exit 1
fi

# --- Environment fixes required on this WSL + pip-installed-CUDA setup ---
# vLLM's FlashInfer sampling kernel JIT-compiles a CUDA source file at first
# request. That build needs `nvcc` and `ninja` on PATH, and a CUDA_HOME with
# matching toolkit headers. pip installs nvcc under nvidia/cu13/bin (no
# system CUDA toolkit is installed in this environment), and `ninja`'s
# console script lives in the venv's own bin/ dir. Even with both present,
# the pip-resolved nvcc (13.2) and FlashInfer's bundled cccl/libcudacxx
# headers were version-skewed and failed to compile, so we disable the
# FlashInfer sampler and let vLLM use its native (non-JIT) top-k/top-p path.
NVIDIA_CU13_DIR="$REPO_ROOT/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME="$NVIDIA_CU13_DIR"
export PATH="$REPO_ROOT/.venv-vllm/bin:$NVIDIA_CU13_DIR/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0

read -r MODEL_ID HOST PORT DTYPE MAX_MODEL_LEN GPU_MEM_UTIL MAX_NUM_SEQS SEED <<EOF
$("$VENV_PYTHON" - "$CONFIG_PATH" <<'PYEOF'
import sys
import yaml

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

print(
    cfg["model_id"], cfg["host"], cfg["port"], cfg["dtype"],
    cfg["max_model_len"], cfg["gpu_memory_utilization"],
    cfg["max_num_seqs"], cfg["seed"],
)
PYEOF
)
EOF

echo "Starting vLLM server"
echo "  config     : $CONFIG_PATH"
echo "  model      : $MODEL_ID"
echo "  host:port  : $HOST:$PORT"
echo "  dtype      : $DTYPE"
echo "  max_model_len            : $MAX_MODEL_LEN"
echo "  gpu_memory_utilization   : $GPU_MEM_UTIL"
echo "  max_num_seqs             : $MAX_NUM_SEQS"

exec "$VLLM_BIN" serve "$MODEL_ID" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --seed "$SEED"
