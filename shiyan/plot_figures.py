from __future__ import annotations

import argparse
from pathlib import Path

from shiyan_benchmark.plotting import plot_exp1, plot_exp2, plot_trace_heatmap


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot shiyan benchmark figures.")
    parser.add_argument("--exp", choices=["exp1", "exp2", "exp4"], required=True)
    parser.add_argument("--summary_csv", type=str, default=None)
    parser.add_argument("--trace_csv", type=str, default=None)
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="If omitted, use shiyan/results/figures/<exp>.png",
    )
    args = parser.parse_args()

    default_output = Path(f"/home/minko/newswa/planC/shiyan/results/figures/{args.exp}.png")
    output_path = args.output_path or str(default_output)

    if args.exp == "exp1":
        summary_csv = args.summary_csv or "/home/minko/newswa/planC/shiyan/results/tables/exp1_summary.csv"
        output = plot_exp1(summary_csv, output_path)
    elif args.exp == "exp2":
        summary_csv = args.summary_csv or "/home/minko/newswa/planC/shiyan/results/tables/exp2_summary.csv"
        output = plot_exp2(summary_csv, output_path)
    else:
        if not args.trace_csv:
            raise ValueError("--trace_csv is required for exp4")
        output = plot_trace_heatmap(args.trace_csv, output_path)
    print(output)


if __name__ == "__main__":
    main()
