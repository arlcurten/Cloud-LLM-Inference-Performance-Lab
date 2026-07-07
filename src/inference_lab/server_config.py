from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ServerConfig:
    model_id: str
    host: str = "127.0.0.1"
    port: int = 8000
    dtype: str = "bfloat16"
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.75
    max_num_seqs: int = 4
    seed: int = 42

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def load_server_config(path: str | Path) -> ServerConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ServerConfig(**data)


def validate_server_config(cfg: ServerConfig) -> None:
    if not (1 <= cfg.port <= 65535):
        raise ValueError(f"port must be between 1 and 65535, got {cfg.port}")
    if not (0.0 < cfg.gpu_memory_utilization <= 1.0):
        raise ValueError(
            f"gpu_memory_utilization must be in (0, 1], got {cfg.gpu_memory_utilization}"
        )
    if cfg.max_model_len <= 0:
        raise ValueError(f"max_model_len must be positive, got {cfg.max_model_len}")
    if cfg.max_num_seqs <= 0:
        raise ValueError(f"max_num_seqs must be positive, got {cfg.max_num_seqs}")
