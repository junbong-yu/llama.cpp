#!/usr/bin/env python3
"""
extract-backend-results.py — Analyze backend-bench JSON output.

Standardized extraction for backend-level comparison results. Supports:
  - Single file: summary table + speedup breakdown
  - Two files:   before/after comparison (e.g. unoptimized vs optimized custom backend)
  - CSV export for spreadsheets / plotting

Usage:
    python3 extract-backend-results.py results.json
    python3 extract-backend-results.py before.json after.json
    python3 extract-backend-results.py results.json --op add
    python3 extract-backend-results.py results.json --format csv > out.csv
    python3 extract-backend-results.py results.json --format markdown
"""

import argparse
import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt_speedup(x: float) -> str:
    if x >= 1.05:
        return f"+{x:.2f}x"
    if x <= 0.95:
        return f"-{x:.2f}x"
    return f"~{x:.2f}x"


def print_single_table(data: dict, op_filter: str | None = None):
    print("=== Backend Benchmark Report ===")
    print(f"Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"Backends : {', '.join(data.get('backends', []))}")
    cfg = data.get("config", {})
    print(f"Config   : min_time={cfg.get('min_time_sec', '?')}s, warmup={cfg.get('warmup', '?')}")
    print()

    # Group results by (op, size)
    results = data["results"]
    if op_filter:
        results = [r for r in results if r["op"] == op_filter]

    # Print per-op group
    seen = set()
    print(f"{'Op':<10} {'Size':<18} {'Backend':<14} {'Avg(us)':>12} "
          f"{'Min(us)':>12} {'BW(GB/s)':>10} {'GFLOPS':>10} {'Runs':>6} {'Match':>6}")
    print("-" * 108)
    for r in sorted(results, key=lambda x: (x["op"], x["size"], x["backend"])):
        bw = f"{r['bandwidth_gb_s']:.1f}" if r["bandwidth_gb_s"] > 0 else "-"
        gf = f"{r['gflops']:.2f}" if r["gflops"] > 0 else "-"
        match = "OK" if r["correct"] else "FAIL"
        print(f"{r['op']:<10} {r['size']:<18} {r['backend']:<14} "
              f"{r['avg_time_us']:>12.1f} {r['min_time_us']:>12.1f} "
              f"{bw:>10} {gf:>10} {r['n_runs']:>6} {match:>6}")
    print()

    # Speedup / comparisons
    comps = data.get("comparisons", [])
    if op_filter:
        comps = [c for c in comps if c["op"] == op_filter]
    if comps:
        print("=== Backend Speedup ===")
        print(f"{'Op':<10} {'Size':<18} {'Baseline':<14} {'Contender':<14} "
              f"{'Baseline(us)':>14} {'Cont(us)':>14} {'Speedup':>10} {'Match':>6}")
        print("-" * 114)
        for c in sorted(comps, key=lambda x: (x["op"], x["size"])):
            match = "OK" if c.get("correctness", True) else "FAIL"
            print(f"{c['op']:<10} {c['size']:<18} {c['baseline_backend']:<14} "
                  f"{c['contender_backend']:<14} {c['baseline_avg_us']:>14.1f} "
                  f"{c['contender_avg_us']:>14.1f} {fmt_speedup(c['speedup']):>10} {match:>6}")
        print()


def print_comparison(before: dict, after: dict, op_filter: str | None = None):
    print("=== Cross-Run Backend Comparison ===")
    print(f"Before : {before.get('timestamp', 'N/A')}")
    print(f"After  : {after.get('timestamp', 'N/A')}")
    print()

    def idx(d):
        return {(r["backend"], r["op"], r["size"]): r for r in d["results"]}

    b_idx = idx(before)
    a_idx = idx(after)
    all_keys = sorted(set(b_idx.keys()) | set(a_idx.keys()))
    if op_filter:
        all_keys = [k for k in all_keys if k[1] == op_filter]

    print(f"{'Backend':<14} {'Op':<10} {'Size':<18} "
          f"{'Before(us)':>12} {'After(us)':>12} {'Change':>10} {'Speedup':>10}")
    print("-" * 100)
    for k in all_keys:
        backend, op, size = k
        b = b_idx.get(k)
        a = a_idx.get(k)
        if b and a and b["avg_time_us"] > 0:
            spd = b["avg_time_us"] / a["avg_time_us"]
            pct = (1 - a["avg_time_us"] / b["avg_time_us"]) * 100
            change = f"{pct:+.1f}%"
            spd_str = fmt_speedup(spd)
        else:
            change = "N/A"
            spd_str = "N/A"
        b_str = f"{b['avg_time_us']:.1f}" if b else "N/A"
        a_str = f"{a['avg_time_us']:.1f}" if a else "N/A"
        print(f"{backend:<14} {op:<10} {size:<18} "
              f"{b_str:>12} {a_str:>12} {change:>10} {spd_str:>10}")
    print()


def print_csv(data: dict, op_filter: str | None = None):
    fields = ["backend", "op", "size", "n_elements", "n_runs",
              "avg_time_us", "min_time_us", "max_time_us",
              "bandwidth_gb_s", "gflops", "correct", "max_abs_err"]
    print(",".join(fields))
    for r in data["results"]:
        if op_filter and r["op"] != op_filter:
            continue
        print(",".join(str(r[f]) for f in fields))


def print_markdown(data: dict, op_filter: str | None = None):
    print(f"# Backend Benchmark — {data.get('timestamp', '')}\n")
    print(f"**Backends:** {', '.join(data.get('backends', []))}  ")
    cfg = data.get("config", {})
    print(f"**Config:** min_time={cfg.get('min_time_sec', '?')}s, "
          f"warmup={cfg.get('warmup', '?')}\n")

    print("## Per-Op Results\n")
    print("| Op | Size | Backend | Avg (us) | Min (us) | BW (GB/s) | GFLOPS | Runs | Match |")
    print("|----|------|---------|---------:|---------:|----------:|-------:|-----:|:-----:|")
    results = data["results"]
    if op_filter:
        results = [r for r in results if r["op"] == op_filter]
    for r in sorted(results, key=lambda x: (x["op"], x["size"], x["backend"])):
        bw = f"{r['bandwidth_gb_s']:.1f}" if r["bandwidth_gb_s"] > 0 else "-"
        gf = f"{r['gflops']:.2f}" if r["gflops"] > 0 else "-"
        match = "OK" if r["correct"] else "FAIL"
        print(f"| {r['op']} | {r['size']} | {r['backend']} | "
              f"{r['avg_time_us']:.1f} | {r['min_time_us']:.1f} | "
              f"{bw} | {gf} | {r['n_runs']} | {match} |")
    print()

    comps = data.get("comparisons", [])
    if op_filter:
        comps = [c for c in comps if c["op"] == op_filter]
    if comps:
        print("## Speedup\n")
        print("| Op | Size | Baseline | Contender | Baseline (us) | Contender (us) | Speedup | Match |")
        print("|----|------|----------|-----------|---------------:|---------------:|--------:|:-----:|")
        for c in sorted(comps, key=lambda x: (x["op"], x["size"])):
            match = "OK" if c.get("correctness", True) else "FAIL"
            print(f"| {c['op']} | {c['size']} | {c['baseline_backend']} | "
                  f"{c['contender_backend']} | {c['baseline_avg_us']:.1f} | "
                  f"{c['contender_avg_us']:.1f} | {fmt_speedup(c['speedup'])} | {match} |")


def main():
    p = argparse.ArgumentParser(description="Extract / analyze backend-bench JSON results")
    p.add_argument("files", nargs="+", help="JSON file(s). 1=single report, 2=comparison")
    p.add_argument("--op", default=None, help="Filter by op name")
    p.add_argument("--format", choices=["table", "csv", "markdown"], default="table")
    args = p.parse_args()

    if len(args.files) == 1:
        d = load(args.files[0])
        if args.format == "csv":
            print_csv(d, args.op)
        elif args.format == "markdown":
            print_markdown(d, args.op)
        else:
            print_single_table(d, args.op)
    elif len(args.files) == 2:
        b = load(args.files[0])
        a = load(args.files[1])
        if args.format != "table":
            print("Comparison mode only supports --format table", file=sys.stderr)
            sys.exit(1)
        print_comparison(b, a, args.op)
    else:
        print("Provide 1 or 2 files", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
