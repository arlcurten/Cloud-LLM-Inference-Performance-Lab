"""Generate simple PNG plots from a benchmark summary CSV."""
import argparse
import csv
import sys
from pathlib import Path

REQUIRED_PLOT_COLS = [
    "experiment_name",
    "input_tokens",
    "mean_prefill_latency_ms",
    "mean_decode_tokens_per_second",
    "mean_e2e_latency_ms",
]


def validate_csv_columns(headers: list[str]) -> None:
    missing = [c for c in REQUIRED_PLOT_COLS if c not in headers]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV file is empty: {path}")
    validate_csv_columns(list(rows[0].keys()))
    return rows


def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def plot_all(csv_path: Path, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [r["experiment_name"] for r in rows]
    x = list(range(len(names)))
    generated_paths = []

    def _save(fig, filename):
        path = out_dir / filename
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        generated_paths.append(path)
        print(f"  Saved: {path}")

    # 1. Prefill latency vs input token count
    input_tok = [_to_float(r["input_tokens"]) for r in rows]
    prefill_ms = [_to_float(r["mean_prefill_latency_ms"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, prefill_ms, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({int(t)} tok)" for n, t in zip(names, input_tok)], fontsize=9)
    ax.set_ylabel("Mean Prefill Latency (ms)")
    ax.set_title("Prefill Latency vs Input Token Count")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save(fig, "prefill_latency_vs_input_tokens.png")

    # 2. Decode tokens per second vs workload
    tps = [_to_float(r["mean_decode_tokens_per_second"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, tps, color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Mean Decode Tokens/s")
    ax.set_title("Decode Throughput vs Workload")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save(fig, "decode_tokens_per_second_vs_workload.png")

    # 3. E2E latency vs workload
    e2e = [_to_float(r["mean_e2e_latency_ms"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, e2e, color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Mean E2E Latency (ms)")
    ax.set_title("End-to-End Latency vs Workload")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save(fig, "e2e_latency_vs_workload.png")

    return generated_paths


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results from CSV")
    parser.add_argument("csv", help="Path to the summary CSV from aggregate_results.py")
    parser.add_argument(
        "--out-dir",
        default="results/plots",
        help="Output directory for PNG files (default: results/plots)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    paths = plot_all(csv_path, Path(args.out_dir))
    print(f"Generated {len(paths)} plot(s).")


if __name__ == "__main__":
    main()
