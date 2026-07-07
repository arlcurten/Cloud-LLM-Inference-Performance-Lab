"""Server-side metadata capture for Phase 2 online inference (vLLM).

This is a metadata utility only — it does not define the full Phase 2
result JSON schema, which will be introduced alongside the Phase 2B
concurrency benchmark.
"""
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .server_config import ServerConfig
from .system_info import collect as collect_system_info

_VLLM_VENV_PYTHON = Path(__file__).resolve().parents[2] / ".venv-vllm" / "bin" / "python"


def collect_server_metadata(server_cfg: ServerConfig) -> dict:
    sysinfo = collect_system_info()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": sysinfo["git_commit"],
        "model_id": server_cfg.model_id,
        "vllm_version": _vllm_version(),
        "pytorch_version": sysinfo["pytorch_version"],
        "cuda_version": sysinfo["cuda_version"],
        "gpu_name": sysinfo["gpu_name"],
        "total_vram_mb": sysinfo["total_vram_mb"],
        "server_config": asdict(server_cfg),
    }


def _vllm_version() -> str:
    if not _VLLM_VENV_PYTHON.exists():
        return "unknown (.venv-vllm not found)"
    try:
        result = subprocess.run(
            [str(_VLLM_VENV_PYTHON), "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"
