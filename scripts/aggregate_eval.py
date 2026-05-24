#!/usr/bin/env python
"""Collect eval_results/<run_name>/{short,long}.json into a single summary CSV.

One row per (run, task, metric) tuple. Use a pivot in a spreadsheet for a per-run
matrix view.

Usage:
    python scripts/aggregate_eval.py --eval-root eval_results --out eval_results/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def load_lm_eval_results(path: Path) -> dict[str, dict[str, float]]:
    """Returns {task_name: {metric_key: value}} (only numeric metrics)."""
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    out: dict[str, dict[str, float]] = {}
    for task, scores in data.get("results", {}).items():
        out[task] = {}
        for k, v in scores.items():
            if isinstance(v, (int, float)):
                out[task][k] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    rows: list[dict[str, str]] = []
    for run_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        run_name = run_dir.name
        for kind in ("short", "long"):
            results = load_lm_eval_results(run_dir / f"{kind}.json")
            for task, metrics in results.items():
                for metric_key, value in metrics.items():
                    rows.append({
                        "run": run_name,
                        "kind": kind,
                        "task": task,
                        "metric": metric_key,
                        "value": f"{value:.6f}",
                    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "kind", "task", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
