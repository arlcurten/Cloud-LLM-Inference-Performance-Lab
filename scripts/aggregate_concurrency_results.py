"""Aggregate a Phase 2B concurrency benchmark JSON file into a summary CSV.

One row per concurrency level. Unlike Phase 1's aggregate_results.py (which
merges many separate benchmark files, one per workload), a single
concurrency benchmark run already contains all concurrency levels in one
JSON file, so this script takes exactly one input file.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

CSV_COLUMNS = [
    "concurrency",
    "request_count",
    "success_count",
    "failure_count",
    "failure_rate",
    "ttft_mean_ms",
    "ttft_median_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "approx_tpot_mean_ms",
    "approx_tpot_median_ms",
    "e2e_mean_ms",
    "e2e_median_ms",
    "e2e_p95_ms",
    "e2e_p99_ms",
    "request_throughput_rps",
    "output_token_throughput_tps",
    "mean_completion_tokens",
    "benchmark_duration_seconds",
]


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_rows(doc: dict) -> list[dict]:
    rows = []
    for experiment in doc["experiments"]:
        summary = experiment["summary"]
        row = {"concurrency": experiment["concurrency"]}
        for col in CSV_COLUMNS:
            if col == "concurrency":
                continue
            row[col] = summary.get(col, "")
        rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate a Phase 2B concurrency benchmark JSON file into a CSV summary"
    )
    parser.add_argument("input", help="Path to a concurrency_benchmark_*.json file")
    parser.add_argument(
        "--output",
        default="results/processed/phase2_concurrency_summary.csv",
        help="Output CSV path (default: results/processed/phase2_concurrency_summary.csv)",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.is_file():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    doc = _load_json(path)
    rows = extract_rows(doc)
    if not rows:
        print("ERROR: no experiments found in input file.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    write_csv(rows, out_path)
    print(f"CSV written to: {out_path}  ({len(rows)} row(s))")


if __name__ == "__main__":
    main()
