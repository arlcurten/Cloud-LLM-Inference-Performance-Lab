"""Smoke-test script: load Gemma and run one deterministic generation."""
import argparse
import sys
from pathlib import Path

import torch

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference_lab.config import load_config
from inference_lab.model_loader import load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Gemma smoke inference")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. A CUDA-capable GPU is required.", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    print(f"Loading model: {cfg.model_id}")
    try:
        tokenizer, model = load_model_and_tokenizer(cfg)
    except Exception as e:
        print(f"ERROR: Failed to load model — {e}", file=sys.stderr)
        sys.exit(1)

    inputs = tokenizer(cfg.prompt, return_tensors="pt").to(cfg.device)
    input_token_count = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
        )

    new_token_count = output_ids.shape[1] - input_token_count
    generated_text = tokenizer.decode(
        output_ids[0, input_token_count:], skip_special_tokens=True
    )

    print(f"\n--- Smoke Inference Result ---")
    print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"Input tokens     : {input_token_count}")
    print(f"Generated tokens : {new_token_count}")
    print(f"Generated text   : {generated_text}")
    print(f"CUDA allocated   : {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"CUDA reserved    : {torch.cuda.memory_reserved() / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
