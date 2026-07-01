import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference_lab.config import InferenceConfig, load_config

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "phase1_smoke.yaml"


def test_load_config_returns_inference_config():
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, InferenceConfig)


def test_load_config_values():
    cfg = load_config(CONFIG_PATH)
    assert cfg.model_id == "google/gemma-3-1b-it"
    assert cfg.model_dir == "models/google/gemma-3-1b-it"
    assert cfg.device == "cuda"
    assert cfg.dtype == "float16"
    assert cfg.max_new_tokens == 32
    assert cfg.do_sample is False
    assert cfg.seed == 42


def test_defaults():
    cfg = InferenceConfig()
    assert cfg.device == "cuda"
    assert cfg.dtype == "float16"
    assert cfg.do_sample is False
    assert cfg.max_new_tokens == 32
    assert cfg.model_dir is None
