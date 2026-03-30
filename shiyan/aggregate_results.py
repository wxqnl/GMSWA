from __future__ import annotations

import argparse

from shiyan_benchmark.aggregation import aggregate_logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate shiyan benchmark run logs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more JSONL run logs.")
    parser.add_argument("--experiment", choices=["exp1", "exp2", "exp3", "exp4"], required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/minko/newswa/planC/shiyan/results/tables",
    )
    args = parser.parse_args()
    outputs = aggregate_logs(
        input_paths=args.inputs,
        experiment=args.experiment,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
