"""Generate simple PNG plots from the Phase 2B concurrency summary CSV."""
import argparse
import csv
import sys
from pathlib import Path

REQUIRED_PLOT_COLS = [
    "concurrency",
    "ttft_median_ms",
    "ttft_p95_ms",
    "e2e_median_ms",
    "e2e_p95_ms",
    "request_throughput_rps",
    "output_token_throughput_tps",
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

    rows = sorted(_load_csv(csv_path), key=lambda r: _to_float(r["concurrency"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    concurrency = [_to_float(r["concurrency"]) for r in rows]
    generated_paths = []

    def _save(fig, filename):
        path = out_dir / filename
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        generated_paths.append(path)
        print(f"  Saved: {path}")

    # 1. TTFT vs concurrency (median + P95)
    ttft_median = [_to_float(r["ttft_median_ms"]) for r in rows]
    ttft_p95 = [_to_float(r["ttft_p95_ms"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(concurrency, ttft_median, marker="o", label="median")
    ax.plot(concurrency, ttft_p95, marker="o", label="p95")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to First Token vs Concurrency")
    ax.set_xticks(concurrency)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save(fig, "phase2_ttft_vs_concurrency.png")

    # 2. E2E latency vs concurrency (median + P95)
    e2e_median = [_to_float(r["e2e_median_ms"]) for r in rows]
    e2e_p95 = [_to_float(r["e2e_p95_ms"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(concurrency, e2e_median, marker="o", label="median")
    ax.plot(concurrency, e2e_p95, marker="o", label="p95")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("E2E Latency (ms)")
    ax.set_title("End-to-End Latency vs Concurrency")
    ax.set_xticks(concurrency)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save(fig, "phase2_e2e_vs_concurrency.png")

    # 3. Throughput vs concurrency (aggregate output tokens/s, requests/s)
    output_tps = [_to_float(r["output_token_throughput_tps"]) for r in rows]
    request_rps = [_to_float(r["request_throughput_rps"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    # linestyles differ (solid vs dashed) because output_tps and request_rps
    # are proportional for a fixed-length workload (constant tokens/request),
    # so on independently auto-scaled twin axes the two curves can coincide
    # pixel-for-pixel — a same-color/marker pair would make one line invisible.
    ax1.plot(concurrency, output_tps, marker="o", linestyle="-", color="darkorange", label="output tokens/s")
    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("Output tokens/s", color="darkorange")
    ax1.set_xticks(concurrency)
    ax2 = ax1.twinx()
    ax2.plot(concurrency, request_rps, marker="s", linestyle="--", color="steelblue", label="requests/s")
    ax2.set_ylabel("Requests/s", color="steelblue")
    ax1.set_title("Throughput vs Concurrency")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    _save(fig, "phase2_throughput_vs_concurrency.png")

    # 4. Latency vs throughput tradeoff: aggregate output tokens/s vs median E2E latency
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(output_tps, e2e_median, marker="o")
    for c, x, y in zip(concurrency, output_tps, e2e_median):
        ax.annotate(f"c={int(c)}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Aggregate Output Tokens/s")
    ax.set_ylabel("Median E2E Latency (ms)")
    ax.set_title("Latency vs Throughput Tradeoff")
    ax.grid(linestyle="--", alpha=0.5)
    _save(fig, "phase2_latency_throughput_tradeoff.png")

    return generated_paths


def main():
    parser = argparse.ArgumentParser(description="Plot Phase 2B concurrency results from CSV")
    parser.add_argument("csv", help="Path to phase2_concurrency_summary.csv")
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
