import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from .config import InferenceConfig

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_model_path(cfg: InferenceConfig) -> str:
    """Return a local model path, downloading if model_dir is configured."""
    if cfg.model_dir is None:
        return cfg.model_id

    local_path = Path(cfg.model_dir)
    # Download flat into model_dir (no nested cache structure)
    snapshot_download(repo_id=cfg.model_id, local_dir=str(local_path))
    return str(local_path)


def load_model_and_tokenizer(cfg: InferenceConfig):
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    dtype = _DTYPE_MAP.get(cfg.dtype)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {cfg.dtype}")

    model_path = _resolve_model_path(cfg)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=cfg.device,
    )
    model.eval()
    return tokenizer, model
