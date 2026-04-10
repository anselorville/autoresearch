"""
Autoresearch experiment analysis.

Reads results.tsv and prints a summary of the experiment run.
Optionally saves a progress chart (progress.png).

Usage:
    python analysis.py                           # use results.tsv in cwd
    python analysis.py --results my_results.tsv  # custom file
    python analysis.py --no-plot                 # skip matplotlib
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# TSV loading (pure stdlib — no pandas dependency)
# ---------------------------------------------------------------------------

def load_results(path: str) -> list[dict]:
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[analysis] File not found: {path}")
        return rows

    if not lines:
        return rows

    header = lines[0].strip().split("\t")
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", maxsplit=len(header) - 1)
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        rows.append(dict(zip(header, parts)))

    return rows


def parse_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _metric_key(rows: list[dict]) -> str:
    """Auto-detect metric column: 'metric' (new pipeline) or 'val_bpb' (legacy)."""
    for r in rows:
        if "metric" in r:
            return "metric"
        if "val_bpb" in r:
            return "val_bpb"
    return "metric"


def print_summary(rows: list[dict]) -> None:
    if not rows:
        print("[analysis] No data to analyze.")
        return

    total = len(rows)
    by_status: dict[str, list[dict]] = {}
    for r in rows:
        s = r.get("status", "").lower()
        by_status.setdefault(s, []).append(r)

    kept    = by_status.get("keep", [])
    discard = by_status.get("discard", [])
    crash   = by_status.get("crash", [])
    mk      = _metric_key(rows)

    print("\n" + "=" * 56)
    print("  Autoresearch Experiment Summary")
    print("=" * 56)
    print(f"  Total runs   : {total}")
    print(f"  Keep         : {len(kept)}")
    print(f"  Discard      : {len(discard)}")
    print(f"  Crash        : {len(crash)}")
    if total > 0:
        print(f"  Keep rate    : {100 * len(kept) / total:.0f}%")

    # Baseline = first keep entry
    if kept:
        values = [parse_float(r.get(mk, "0")) for r in kept
                  if parse_float(r.get(mk, "0")) > 0]
        if values:
            baseline_val = values[0]
            best_val     = min(values)
            best_row     = next(r for r in kept
                                if parse_float(r.get(mk, "0")) == best_val)
            improvement  = (baseline_val - best_val) / baseline_val * 100

            print()
            print(f"  Baseline     : {baseline_val:.6f}")
            print(f"  Best         : {best_val:.6f}  (run: {best_row.get('commit', '?')})")
            print(f"  Improvement  : {improvement:.2f}%")
            print(f"  Best exp.    : {best_row.get('description', '?')}")

    # All KEEP experiments in order
    print()
    print("  Kept experiments:")
    print(f"  {'#':>3}  {'commit':>9}  {mk:>12}  {'mem(GB)':>8}  description")
    print("  " + "-" * 56)
    for i, r in enumerate(kept, 1):
        val  = parse_float(r.get(mk, "0"))
        mem  = parse_float(r.get("memory_gb", "0"))
        cmt  = r.get("commit", "?")[:9]
        desc = r.get("description", "")[:38]
        baseline_marker = " (baseline)" if i == 1 else ""
        print(f"  {i:>3}  {cmt:>9}  {val:12.6f}  {mem:8.1f}  {desc}{baseline_marker}")

    # Top improving steps
    if len(kept) >= 3:
        deltas     = []
        val_series = [parse_float(r.get(mk, "0")) for r in kept
                      if parse_float(r.get(mk, "0")) > 0]
        for idx in range(1, len(val_series)):
            delta = val_series[idx - 1] - val_series[idx]
            if delta > 0:
                deltas.append((delta, kept[idx].get("description", "?")))
        if deltas:
            deltas.sort(reverse=True)
            print()
            print("  Top improvements:")
            for delta, desc in deltas[:5]:
                print(f"    Δ={delta:.6f}  {desc[:50]}")

    print("=" * 56)


# ---------------------------------------------------------------------------
# Progress chart
# ---------------------------------------------------------------------------

def save_progress_chart(rows: list[dict], out_path: str = "progress.png") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analysis] matplotlib not available — skipping chart.")
        return

    if not rows:
        return

    mk       = _metric_key(rows)
    all_vals = [parse_float(r.get(mk, "0")) for r in rows]
    statuses = [r.get("status", "").lower() for r in rows]

    # Running best
    best_so_far  = []
    current_best = float("inf")
    for val, status in zip(all_vals, statuses):
        if status == "keep" and val > 0:
            current_best = min(current_best, val)
        best_so_far.append(current_best if current_best != float("inf") else None)

    fig, ax = plt.subplots(figsize=(10, 5))

    xs = list(range(1, len(rows) + 1))

    # Scatter: color by status
    colors = {"keep": "#2ecc71", "discard": "#e74c3c", "crash": "#95a5a6"}
    for status in ("keep", "discard", "crash"):
        xs_s   = [x for x, s, v in zip(xs, statuses, all_vals) if s == status and v > 0]
        vals_s = [v for s, v in zip(statuses, all_vals) if s == status and v > 0]
        if xs_s:
            ax.scatter(xs_s, vals_s, c=colors.get(status, "#7f8c8d"),
                       label=status.capitalize(), zorder=3, s=60, alpha=0.8)

    # Running best step line
    xs_best   = [x for x, b in zip(xs, best_so_far) if b is not None]
    vals_best = [b for b in best_so_far if b is not None]
    if xs_best:
        ax.step(xs_best, vals_best, where="post", color="#2980b9",
                linewidth=2, label="Running best", zorder=2)

    ax.set_xlabel("Experiment #")
    ax.set_ylabel(f"{mk} (lower is better)")
    ax.set_title("Autoresearch Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if vals_best:
        margin = max(0.01, (max(vals_best) - min(vals_best)) * 0.3 + 0.01)
        ax.set_ylim(min(vals_best) - margin, max(vals_best) + margin * 3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[analysis] Chart saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze autoresearch results")
    parser.add_argument("--results", default="results.tsv",
                        help="Path to results TSV file (default: results.tsv)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip saving progress.png")
    parser.add_argument("--output", default="progress.png",
                        help="Output chart path (default: progress.png)")
    args = parser.parse_args()

    rows = load_results(args.results)
    print_summary(rows)

    if not args.no_plot:
        save_progress_chart(rows, out_path=args.output)


if __name__ == "__main__":
    main()
