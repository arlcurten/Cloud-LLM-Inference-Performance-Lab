"""Aggregate multiple benchmark JSON files into a single CSV summary."""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CSV_COLUMNS = [
    "experiment_name",
    "model_id",
    "dtype",
    "input_tokens",
    "requested_output_tokens",
    "mean_generated_tokens",
    "mean_prefill_latency_ms",
    "median_prefill_latency_ms",
    "mean_decode_token_latency_ms",
    "median_decode_token_latency_ms",
    "mean_decode_tokens_per_second",
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "peak_cuda_allocated_mb",
    "peak_cuda_reserved_mb",
]


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _extract_row(doc: dict, path: Path) -> dict:
    meta = doc["metadata"]
    conf = doc["configuration"]
    summ = doc["summary"]

    experiment_name = meta.get("experiment_name") or path.stem

    def _stat(key, stat):
        val = summ.get(key)
        if val is None:
            return ""
        if isinstance(val, dict):
            return val.get(stat, "")
        # scalar (e.g. input_tokens)
        return val

    return {
        "experiment_name": experiment_name,
        "model_id": meta.get("model_id", ""),
        "dtype": meta.get("dtype", ""),
        "input_tokens": _stat("input_tokens", "mean"),
        "requested_output_tokens": conf.get("max_new_tokens", ""),
        "mean_generated_tokens": _stat("generated_tokens", "mean"),
        "mean_prefill_latency_ms": _stat("prefill_latency_ms", "mean"),
        "median_prefill_latency_ms": _stat("prefill_latency_ms", "median"),
        "mean_decode_token_latency_ms": _stat("mean_decode_token_latency_ms", "mean"),
        "median_decode_token_latency_ms": _stat("mean_decode_token_latency_ms", "median"),
        "mean_decode_tokens_per_second": _stat("decode_tokens_per_second", "mean"),
        "mean_e2e_latency_ms": _stat("e2e_latency_ms", "mean"),
        "median_e2e_latency_ms": _stat("e2e_latency_ms", "median"),
        "peak_cuda_allocated_mb": _stat("peak_cuda_allocated_mb", "max"),
        "peak_cuda_reserved_mb": _stat("peak_cuda_reserved_mb", "max"),
    }


def collect_json_files(inputs: list[str]) -> list[Path]:
    files = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        elif p.is_file() and p.suffix == ".json":
            files.append(p)
        else:
            print(f"WARNING: skipping {inp} (not a JSON file or directory)", file=sys.stderr)
    return files


def aggregate(json_files: list[Path]) -> list[dict]:
    rows = []
    for path in json_files:
        try:
            doc = _load_json(path)
            row = _extract_row(doc, path)
            rows.append(row)
            print(f"  Loaded: {path.name}  →  {row['experiment_name']}")
        except Exception as e:
            print(f"WARNING: skipping {path.name}: {e}", file=sys.stderr)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark JSON files into a CSV summary"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSON files or a directory containing JSON files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: results/processed/summary_<timestamp>.csv)",
    )
    args = parser.parse_args()

    json_files = collect_json_files(args.inputs)
    if not json_files:
        print("ERROR: no JSON files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Aggregating {len(json_files)} file(s)...")
    rows = aggregate(json_files)

    if not rows:
        print("ERROR: no rows could be extracted.", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = Path("results/processed") / f"summary_{ts}.csv"

    write_csv(rows, out_path)
    print(f"CSV written to: {out_path}  ({len(rows)} row(s))")


if __name__ == "__main__":
    main()
