#!/usr/bin/env python3
"""
compare-kernels.py — ggml Kernel Benchmark Result Analyzer

Usage:
    python3 compare-kernels.py results.json                   # Single file analysis
    python3 compare-kernels.py before.json after.json         # Compare two runs
    python3 compare-kernels.py results.json --op add          # Filter by op
    python3 compare-kernels.py results.json --format csv      # CSV output
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_single_report(data: dict, op_filter: str = None):
    """Print analysis of a single benchmark result file."""
    print(f"=== Kernel Benchmark Report ===")
    print(f"Timestamp : {data.get('timestamp', 'N/A')}")
    print(f"System    : {data['system_info']['os']} / {data['system_info']['arch']}")
    print(f"Config    : warmup={data['config']['warmup_runs']}, min_time={data['config']['min_time_sec']}s")
    print()

    results = data["results"]
    if op_filter:
        results = [r for r in results if r["op"] == op_filter]

    if not results:
        print("No results found.")
        return

    # Group by op
    ops = sorted(set(r["op"] for r in results))

    for op in ops:
        op_results = [r for r in results if r["op"] == op]
        print(f"--- {op.upper()} ---")
        print(f"{'Size':<18} {'Variant':<12} {'Avg(us)':>12} {'Min(us)':>12} "
              f"{'Stddev':>10} {'Runs':>6} {'BW(GB/s)':>10} {'GFLOPS':>10} {'OK':>4}")

        for r in sorted(op_results, key=lambda x: (x["size"], x["variant"])):
            bw = f"{r['bandwidth_gb_s']:.1f}" if r["bandwidth_gb_s"] > 0 else "-"
            gf = f"{r['gflops']:.2f}" if r["gflops"] > 0 else "-"
            ok = "OK" if r["correctness"] else "FAIL"
            print(f"{r['size']:<18} {r['variant']:<12} {r['avg_time_us']:>12.1f} "
                  f"{r['min_time_us']:>12.1f} {r['stddev_time_us']:>10.1f} "
                  f"{r['n_runs']:>6} {bw:>10} {gf:>10} {ok:>4}")
        print()

    # Comparisons
    comparisons = data.get("comparisons", [])
    if op_filter:
        comparisons = [c for c in comparisons if c["op"] == op_filter]

    if comparisons:
        print("=== Speedup Summary ===")
        print(f"{'Op':<12} {'Size':<18} {'Baseline(us)':>14} {'Contender(us)':>14} {'Speedup':>10}")
        for c in sorted(comparisons, key=lambda x: (x["op"], x["size"])):
            speedup_str = f"{c['speedup']:.2f}x"
            if c["speedup"] > 1.05:
                speedup_str = f"+{speedup_str}"
            elif c["speedup"] < 0.95:
                speedup_str = f" {speedup_str}"
            else:
                speedup_str = f"~{speedup_str}"
            print(f"{c['op']:<12} {c['size']:<18} {c['baseline_avg_us']:>14.1f} "
                  f"{c['contender_avg_us']:>14.1f} {speedup_str:>10}")
        print()


def print_comparison_report(before: dict, after: dict, op_filter: str = None):
    """Compare two benchmark runs (e.g. before/after optimization)."""
    print("=== Cross-Run Comparison ===")
    print(f"Before: {before.get('timestamp', 'N/A')}")
    print(f"After : {after.get('timestamp', 'N/A')}")
    print()

    # Index results by (op, variant, size)
    def index_results(data):
        idx = {}
        for r in data["results"]:
            key = (r["op"], r["variant"], r["size"])
            idx[key] = r
        return idx

    before_idx = index_results(before)
    after_idx = index_results(after)

    all_keys = sorted(set(before_idx.keys()) | set(after_idx.keys()))
    if op_filter:
        all_keys = [k for k in all_keys if k[0] == op_filter]

    print(f"{'Op':<12} {'Variant':<12} {'Size':<18} {'Before(us)':>12} {'After(us)':>12} "
          f"{'Change':>10} {'Speedup':>10}")

    for key in all_keys:
        op, variant, size = key
        b = before_idx.get(key)
        a = after_idx.get(key)

        b_avg = b["avg_time_us"] if b else float("nan")
        a_avg = a["avg_time_us"] if a else float("nan")

        if b and a and b_avg > 0:
            speedup = b_avg / a_avg
            pct = (1 - a_avg / b_avg) * 100
            change = f"{pct:+.1f}%"
            spd = f"{speedup:.2f}x"
        else:
            change = "N/A"
            spd = "N/A"

        b_str = f"{b_avg:.1f}" if b else "N/A"
        a_str = f"{a_avg:.1f}" if a else "N/A"

        print(f"{op:<12} {variant:<12} {size:<18} {b_str:>12} {a_str:>12} "
              f"{change:>10} {spd:>10}")

    print()


def print_csv(data: dict, op_filter: str = None):
    """Output results as CSV."""
    results = data["results"]
    if op_filter:
        results = [r for r in results if r["op"] == op_filter]

    fields = ["op", "variant", "size", "n_elements", "n_runs",
              "avg_time_us", "min_time_us", "max_time_us", "stddev_time_us",
              "bandwidth_gb_s", "gflops", "correctness", "max_abs_error", "max_rel_error"]
    print(",".join(fields))
    for r in results:
        vals = [str(r[f]) for f in fields]
        print(",".join(vals))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ggml kernel benchmark results")
    parser.add_argument("files", nargs="+", help="JSON result file(s)")
    parser.add_argument("--op", default=None, help="Filter by op name")
    parser.add_argument("--format", choices=["table", "csv"], default="table",
                        help="Output format (default: table)")
    args = parser.parse_args()

    if len(args.files) == 1:
        data = load_results(args.files[0])
        if args.format == "csv":
            print_csv(data, args.op)
        else:
            print_single_report(data, args.op)
    elif len(args.files) == 2:
        before = load_results(args.files[0])
        after = load_results(args.files[1])
        if args.format == "csv":
            print("CSV format not supported for comparison mode.", file=sys.stderr)
            sys.exit(1)
        print_comparison_report(before, after, args.op)
    else:
        print("Provide 1 file (single report) or 2 files (comparison).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
