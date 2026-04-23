#!/usr/bin/env python3
"""
run-ab-bench.py -- Multi-seed A/B benchmark orchestrator.

Drives `llama-kernel-bench` (micro) or `llama-inference-bench` (live model)
across N seeds per variant, aggregates per-(op,variant,size) mean/stddev,
and reports speedup with a variance-aware significance flag.

Why this exists:
  scripts/compare-kernels.py compares exactly one or two JSON files and has
  no multi-seed aggregation. Kernel experiments on shared hardware have
  enough noise that single-seed speedup numbers routinely mislead. This
  script runs each variant multiple times, averages, and marks only the
  results that are outside the noise floor.

Usage:
  # Kernel micro-bench: compare baseline "standard" vs your "sve" variant
  python3 scripts/run-ab-bench.py kernel \
      --bin build-arm/bin/llama-kernel-bench \
      --variants standard sve \
      --ops relu silu \
      --seeds 5 \
      --out-dir /tmp/ab_kernel

  # Inference bench: standard vs custom hook, N seeds
  python3 scripts/run-ab-bench.py inference \
      --bin build-arm/bin/llama-inference-bench \
      --model models/Qwen3-0.6B-Q8_0.gguf \
      --pp 256 --tg 64 --threads 4 \
      --seeds 3 \
      --out-dir /tmp/ab_infer
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


# ----------------------------- stats helpers ------------------------------- #

def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.fmean(xs), statistics.stdev(xs)


def cv_pct(mean, std):
    if mean == 0 or math.isnan(mean):
        return float("nan")
    return 100.0 * std / abs(mean)


def significance_flag(speedup, cv_baseline_pct, cv_user_pct, threshold=0.03):
    """Return one of: 'IMPROVED', 'REGRESSED', 'NOISE', 'NOISY_MEASUREMENT'."""
    noise_threshold = max(cv_baseline_pct, cv_user_pct) / 100.0
    if noise_threshold > 0.05:
        return "NOISY_MEASUREMENT"  # CV > 5% — can't tell
    if speedup > 1.0 + threshold and (speedup - 1.0) > 2 * noise_threshold:
        return "IMPROVED"
    if speedup < 1.0 - threshold and (1.0 - speedup) > 2 * noise_threshold:
        return "REGRESSED"
    return "NOISE"


# ----------------------------- runner core -------------------------------- #

def run_once(cmd, label, seed_idx):
    t0 = time.time()
    print(f"  [seed {seed_idx}] {' '.join(cmd)}", flush=True)
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after 1800s", file=sys.stderr)
        return None, None
    dt = time.time() - t0
    if cp.returncode != 0:
        print(f"    FAIL (exit {cp.returncode}) in {dt:.1f}s", file=sys.stderr)
        print(cp.stderr[-1000:], file=sys.stderr)
        return None, None
    print(f"    OK in {dt:.1f}s", flush=True)
    return cp.stdout, cp.stderr


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------- kernel mode -------------------------------- #

def kernel_mode(args):
    """Run llama-kernel-bench N times; filter to requested variants/ops;
    aggregate per (op, variant, size)."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # llama-kernel-bench runs all variants in one invocation, so we only need
    # N seeds total (not N * |variants|). The --variant flag filters output.
    seed_files = []
    for s in range(1, args.seeds + 1):
        out = out_dir / f"kernel_seed{s}.json"
        cmd = [args.bin, "--output", str(out),
               "--min-time", str(args.min_time),
               "--warmup", str(args.warmup)]
        if args.ops:
            # kernel-bench --op is singular; if multi-op filter requested we
            # drop it (all ops present) and filter in post-processing below.
            if len(args.ops) == 1:
                cmd += ["--op", args.ops[0]]
        stdout, stderr = run_once(cmd, f"kernel seed{s}", s)
        if stdout is None:
            return 1
        seed_files.append(out)

    # Aggregate: index[op, variant, size] -> list of avg_time_us
    bucket = {}
    correctness = {}
    for path in seed_files:
        data = load_json(path)
        for r in data["results"]:
            if args.ops and r["op"] not in args.ops:
                continue
            if args.variants and r["variant"] not in args.variants:
                continue
            key = (r["op"], r["variant"], r["size"])
            bucket.setdefault(key, []).append(r["avg_time_us"])
            correctness.setdefault(key, []).append(r["correctness"])

    # Print per-variant summary
    print("\n## Per-variant timings (aggregated across seeds)\n")
    print(f"{'op':<10} {'variant':<10} {'size':<16} {'mean(us)':>12} {'std':>10} {'cv%':>7} {'seeds':>6} {'correct':>8}")
    for key in sorted(bucket):
        op, variant, size = key
        xs = bucket[key]
        m, s = mean_std(xs)
        cv = cv_pct(m, s)
        ok = "OK" if all(correctness[key]) else "FAIL"
        print(f"{op:<10} {variant:<10} {size:<16} {m:>12.2f} {s:>10.2f} {cv:>7.2f} {len(xs):>6} {ok:>8}")

    # Pairwise speedup: for each op x size, take baseline (first variant) vs others
    if len(args.variants) >= 2:
        baseline = args.variants[0]
        print(f"\n## Speedup vs baseline variant='{baseline}'\n")
        print(f"{'op':<10} {'variant':<10} {'size':<16} {'speedup':>10} {'verdict':>18}")
        ops_sizes = sorted({(k[0], k[2]) for k in bucket})
        for op, size in ops_sizes:
            b_key = (op, baseline, size)
            if b_key not in bucket:
                continue
            b_xs = bucket[b_key]
            b_mean, b_std = mean_std(b_xs)
            b_cv = cv_pct(b_mean, b_std)
            for v in args.variants[1:]:
                u_key = (op, v, size)
                if u_key not in bucket:
                    print(f"{op:<10} {v:<10} {size:<16} {'N/A':>10} {'MISSING':>18}")
                    continue
                u_xs = bucket[u_key]
                u_mean, u_std = mean_std(u_xs)
                u_cv = cv_pct(u_mean, u_std)
                speedup = b_mean / u_mean if u_mean > 0 else float("nan")
                verdict = significance_flag(speedup, b_cv, u_cv,
                                            threshold=args.threshold)
                print(f"{op:<10} {v:<10} {size:<16} {speedup:>9.3f}x {verdict:>18}")

    # Save aggregated summary next to raw files
    summary = {
        "mode": "kernel",
        "seeds": args.seeds,
        "variants": args.variants,
        "ops": args.ops,
        "threshold": args.threshold,
        "buckets": [
            {"op": k[0], "variant": k[1], "size": k[2],
             "mean_us": mean_std(v)[0], "std_us": mean_std(v)[1],
             "n_seeds": len(v), "all_correct": all(correctness[k])}
            for k, v in sorted(bucket.items())
        ],
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAggregated summary: {summary_path}")
    return 0


# --------------------------- inference mode ------------------------------- #

def inference_mode(args):
    """Run llama-inference-bench N times; aggregate per_op and pp/tg."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_files = []
    for s in range(1, args.seeds + 1):
        out = out_dir / f"inference_seed{s}.json"
        cmd = [args.bin,
               "-m", args.model,
               "-pp", str(args.pp),
               "-tg", str(args.tg),
               "-t", str(args.threads),
               "-r", "1",
               "-o", str(out)]
        stdout, stderr = run_once(cmd, f"inference seed{s}", s)
        if stdout is None:
            return 1
        seed_files.append(out)

    # Aggregate top-level pp/tg and per_op
    pp_std, pp_cust = [], []
    tg_std, tg_cust = [], []
    per_op = {}  # op -> {std_us: [...], cust_us: [...], calls: [...]}
    for path in seed_files:
        data = load_json(path)
        inf = data.get("inference", {})
        if "standard" in inf:
            pp_std.append(inf["standard"].get("pp_tokens_s", float("nan")))
            tg_std.append(inf["standard"].get("tg_tokens_s", float("nan")))
        if "custom" in inf:
            pp_cust.append(inf["custom"].get("pp_tokens_s", float("nan")))
            tg_cust.append(inf["custom"].get("tg_tokens_s", float("nan")))
        for op_rec in data.get("per_op", []):
            name = op_rec["op"]
            b = per_op.setdefault(name, {"std": [], "cust": [], "calls": []})
            b["std"].append(op_rec.get("std_total_us", float("nan")))
            b["cust"].append(op_rec.get("custom_total_us", float("nan")))
            b["calls"].append(op_rec.get("calls", 0))

    print("\n## Model-level throughput (tokens/s)\n")
    print(f"{'metric':<10} {'standard_mean':>14} {'std':>10} {'custom_mean':>14} {'std':>10} {'speedup':>10} {'verdict':>18}")
    for metric, s_xs, c_xs in [("pp", pp_std, pp_cust), ("tg", tg_std, tg_cust)]:
        s_m, s_s = mean_std(s_xs)
        c_m, c_s = mean_std(c_xs)
        s_cv = cv_pct(s_m, s_s)
        c_cv = cv_pct(c_m, c_s)
        spd = c_m / s_m if s_m else float("nan")  # throughput: higher=faster
        verdict = significance_flag(spd, s_cv, c_cv, threshold=args.threshold)
        print(f"{metric:<10} {s_m:>14.2f} {s_s:>10.2f} {c_m:>14.2f} {c_s:>10.2f} {spd:>9.3f}x {verdict:>18}")

    if args.ops:
        per_op_keys = [k for k in sorted(per_op) if k in args.ops]
    else:
        per_op_keys = sorted(per_op)

    print("\n## Per-op timings (standard_total vs custom_total)\n")
    print(f"{'op':<12} {'std_mean_us':>14} {'cust_mean_us':>14} {'std_cv%':>8} {'cust_cv%':>8} {'speedup':>10} {'verdict':>18}")
    for op in per_op_keys:
        b = per_op[op]
        s_m, s_s = mean_std(b["std"])
        c_m, c_s = mean_std(b["cust"])
        s_cv = cv_pct(s_m, s_s)
        c_cv = cv_pct(c_m, c_s)
        # lower time = faster, so speedup = std / cust
        spd = s_m / c_m if c_m else float("nan")
        verdict = significance_flag(spd, s_cv, c_cv, threshold=args.threshold)
        print(f"{op:<12} {s_m:>14.2f} {c_m:>14.2f} {s_cv:>8.2f} {c_cv:>8.2f} {spd:>9.3f}x {verdict:>18}")

    summary = {
        "mode": "inference",
        "seeds": args.seeds,
        "model": args.model,
        "config": {"pp": args.pp, "tg": args.tg, "threads": args.threads},
        "threshold": args.threshold,
        "throughput": {
            "pp_standard": {"mean": mean_std(pp_std)[0], "std": mean_std(pp_std)[1]},
            "pp_custom":   {"mean": mean_std(pp_cust)[0], "std": mean_std(pp_cust)[1]},
            "tg_standard": {"mean": mean_std(tg_std)[0], "std": mean_std(tg_std)[1]},
            "tg_custom":   {"mean": mean_std(tg_cust)[0], "std": mean_std(tg_cust)[1]},
        },
        "per_op": [
            {"op": op,
             "std_mean_us": mean_std(per_op[op]["std"])[0],
             "std_std_us":  mean_std(per_op[op]["std"])[1],
             "cust_mean_us": mean_std(per_op[op]["cust"])[0],
             "cust_std_us":  mean_std(per_op[op]["cust"])[1],
             "calls_median": statistics.median(per_op[op]["calls"])}
            for op in sorted(per_op)
        ],
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAggregated summary: {summary_path}")
    return 0


# ----------------------------- arg parser --------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(
        description="Multi-seed A/B orchestrator for kernel-bench / inference-bench.")
    sub = p.add_subparsers(dest="mode", required=True)

    k = sub.add_parser("kernel", help="Run llama-kernel-bench with multi-seed aggregation.")
    k.add_argument("--bin", default="build-arm/bin/llama-kernel-bench",
                   help="Path to llama-kernel-bench binary.")
    k.add_argument("--variants", nargs="+", default=["standard", "custom"],
                   help="Variants to compare. First is treated as baseline.")
    k.add_argument("--ops", nargs="*", default=None,
                   help="Filter ops (e.g. relu silu gelu). Empty = all.")
    k.add_argument("--seeds", type=int, default=3, help="Number of runs per variant.")
    k.add_argument("--warmup", type=int, default=3)
    k.add_argument("--min-time", type=float, default=1.0)
    k.add_argument("--threshold", type=float, default=0.03,
                   help="Speedup threshold for IMPROVED/REGRESSED verdict.")
    k.add_argument("--out-dir", default="/tmp/ab_kernel")

    i = sub.add_parser("inference", help="Run llama-inference-bench with multi-seed aggregation.")
    i.add_argument("--bin", default="build-arm/bin/llama-inference-bench")
    i.add_argument("--model", required=True, help="Path to .gguf model file.")
    i.add_argument("--pp", type=int, default=256)
    i.add_argument("--tg", type=int, default=64)
    i.add_argument("--threads", type=int, default=4)
    i.add_argument("--seeds", type=int, default=3)
    i.add_argument("--ops", nargs="*", default=None,
                   help="Filter per_op report (default: all).")
    i.add_argument("--threshold", type=float, default=0.03)
    i.add_argument("--out-dir", default="/tmp/ab_infer")

    return p


def main():
    args = build_parser().parse_args()
    if not os.access(args.bin, os.X_OK):
        print(f"ERROR: binary not found or not executable: {args.bin}", file=sys.stderr)
        return 2
    if args.mode == "kernel":
        return kernel_mode(args)
    if args.mode == "inference":
        if not os.path.exists(args.model):
            print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
            return 2
        return inference_mode(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
