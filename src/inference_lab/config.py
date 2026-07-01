from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class InferenceConfig:
    model_id: str = "google/gemma-3-1b-it"
    device: str = "cuda"
    dtype: str = "float16"
    prompt: str = "Hello, world!"
    max_new_tokens: int = 32
    do_sample: bool = False
    seed: int = 42
    # If set, model is downloaded to this local path instead of the HF cache
    model_dir: Optional[str] = None


def load_config(path: str | Path) -> InferenceConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return InferenceConfig(**data)
