import subprocess
import sys
import torch
import transformers


def collect() -> dict:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    total_vram_mb = (
        torch.cuda.get_device_properties(0).total_memory / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )
    return {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "gpu_name": gpu_name,
        "total_vram_mb": round(total_vram_mb, 1),
        "git_commit": _git_commit(),
    }


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"
