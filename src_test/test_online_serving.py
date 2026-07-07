"""Tests for Phase 2A online serving config, clients, and metadata (no GPU/server required)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from inference_lab.server_config import ServerConfig, load_server_config, validate_server_config
from inference_lab.server_info import collect_server_metadata

from run_online_smoke import build_url, extract_completion_text
from run_streaming_smoke import parse_sse_line, extract_text_delta, compute_streaming_metrics


# ── ServerConfig ──────────────────────────────────────────────────────────────

def test_server_config_defaults():
    cfg = ServerConfig(model_id="models/google/gemma-3-1b-it")
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8000
    assert cfg.base_url == "http://127.0.0.1:8000"


def test_load_server_config_from_yaml(tmp_path):
    yaml_path = tmp_path / "server.yaml"
    yaml_path.write_text(
        "model_id: models/google/gemma-3-1b-it\n"
        "host: 0.0.0.0\n"
        "port: 9000\n"
        "dtype: float16\n"
        "max_model_len: 2048\n"
        "gpu_memory_utilization: 0.75\n"
        "max_num_seqs: 4\n"
        "seed: 42\n"
    )
    cfg = load_server_config(yaml_path)
    assert cfg.model_id == "models/google/gemma-3-1b-it"
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9000


def test_validate_server_config_passes():
    cfg = ServerConfig(model_id="m")
    validate_server_config(cfg)  # should not raise


def test_validate_server_config_bad_port():
    cfg = ServerConfig(model_id="m", port=0)
    with pytest.raises(ValueError, match="port"):
        validate_server_config(cfg)


def test_validate_server_config_bad_port_too_high():
    cfg = ServerConfig(model_id="m", port=70000)
    with pytest.raises(ValueError, match="port"):
        validate_server_config(cfg)


def test_validate_server_config_bad_gpu_memory_utilization():
    cfg = ServerConfig(model_id="m", gpu_memory_utilization=1.5)
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        validate_server_config(cfg)


def test_validate_server_config_zero_gpu_memory_utilization():
    cfg = ServerConfig(model_id="m", gpu_memory_utilization=0.0)
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        validate_server_config(cfg)


def test_validate_server_config_bad_max_model_len():
    cfg = ServerConfig(model_id="m", max_model_len=0)
    with pytest.raises(ValueError, match="max_model_len"):
        validate_server_config(cfg)


def test_validate_server_config_bad_max_num_seqs():
    cfg = ServerConfig(model_id="m", max_num_seqs=-1)
    with pytest.raises(ValueError, match="max_num_seqs"):
        validate_server_config(cfg)


# ── server_info ───────────────────────────────────────────────────────────────

def test_collect_server_metadata_has_required_keys():
    cfg = ServerConfig(model_id="models/google/gemma-3-1b-it")
    meta = collect_server_metadata(cfg)
    required_keys = [
        "timestamp", "git_commit", "model_id", "vllm_version",
        "pytorch_version", "cuda_version", "gpu_name", "total_vram_mb",
        "server_config",
    ]
    for key in required_keys:
        assert key in meta, f"Missing metadata key: {key}"
    assert meta["server_config"]["model_id"] == "models/google/gemma-3-1b-it"
    assert isinstance(meta["vllm_version"], str)


# ── URL construction ──────────────────────────────────────────────────────────

def test_build_url_no_trailing_slash():
    assert build_url("http://127.0.0.1:8000", "/v1/models") == "http://127.0.0.1:8000/v1/models"


def test_build_url_with_trailing_slash_on_base():
    assert build_url("http://127.0.0.1:8000/", "/v1/models") == "http://127.0.0.1:8000/v1/models"


def test_build_url_path_without_leading_slash():
    assert build_url("http://127.0.0.1:8000", "v1/completions") == "http://127.0.0.1:8000/v1/completions"


# ── non-streaming response parsing / error handling ──────────────────────────

def test_extract_completion_text_valid():
    body = {"choices": [{"text": "Paris"}]}
    assert extract_completion_text(body) == "Paris"


def test_extract_completion_text_missing_choices_raises():
    with pytest.raises(ValueError, match="choices"):
        extract_completion_text({})


def test_extract_completion_text_empty_choices_raises():
    with pytest.raises(ValueError, match="choices"):
        extract_completion_text({"choices": []})


def test_extract_completion_text_missing_text_raises():
    with pytest.raises(ValueError, match="text"):
        extract_completion_text({"choices": [{}]})


# ── SSE parsing ────────────────────────────────────────────────────────────────

def test_parse_sse_line_valid_data():
    chunk = parse_sse_line('data: {"choices": [{"text": "hi"}]}')
    assert chunk == {"choices": [{"text": "hi"}]}


def test_parse_sse_line_done_marker():
    assert parse_sse_line("data: [DONE]") is None


def test_parse_sse_line_blank():
    assert parse_sse_line("") is None


def test_parse_sse_line_non_data_line():
    assert parse_sse_line(": comment") is None


# ── text delta extraction ─────────────────────────────────────────────────────

def test_extract_text_delta_completions_shape():
    chunk = {"choices": [{"text": "Paris"}]}
    assert extract_text_delta(chunk) == "Paris"


def test_extract_text_delta_chat_shape():
    chunk = {"choices": [{"delta": {"content": "Paris"}}]}
    assert extract_text_delta(chunk) == "Paris"


def test_extract_text_delta_role_only_chat_delta():
    chunk = {"choices": [{"delta": {"role": "assistant"}}]}
    assert extract_text_delta(chunk) == ""


def test_extract_text_delta_no_choices():
    assert extract_text_delta({"choices": []}) == ""


# ── streaming timing calculation (mocked chunk arrival times) ────────────────

def test_compute_streaming_metrics_basic():
    request_start = 100.0
    chunk_times = [100.05, 100.08, 100.12]
    metrics = compute_streaming_metrics(request_start, chunk_times)
    assert metrics["ttft_ms"] == pytest.approx(50.0)
    assert metrics["e2e_latency_ms"] == pytest.approx(120.0)
    assert metrics["num_chunks"] == 3
    assert metrics["inter_chunk_latencies_ms"] == pytest.approx([30.0, 40.0])


def test_compute_streaming_metrics_single_chunk():
    metrics = compute_streaming_metrics(100.0, [100.02])
    assert metrics["ttft_ms"] == pytest.approx(20.0)
    assert metrics["e2e_latency_ms"] == pytest.approx(20.0)
    assert metrics["inter_chunk_latencies_ms"] == []


def test_compute_streaming_metrics_no_chunks_raises():
    with pytest.raises(ValueError, match="No content chunks"):
        compute_streaming_metrics(100.0, [])
