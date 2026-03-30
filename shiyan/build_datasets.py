from __future__ import annotations

import argparse

from shiyan_benchmark.dataset_builder import build_demo_datasets, build_real_suite_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real suite manifests or demo datasets for the shiyan benchmark.")
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/minko/newswa/planC/shiyan",
        help="Root directory that contains data/processed.",
    )
    parser.add_argument(
        "--mode",
        choices=["real", "demo", "all"],
        default="real",
        help="real: official benchmark suite manifests; demo: old smoke-test JSONL; all: both",
    )
    args = parser.parse_args()
    if args.mode in {"real", "all"}:
        paths = build_real_suite_manifests(args.output_root)
        for name, path in paths.items():
            print(f"real_suite::{name}: {path}")
    if args.mode in {"demo", "all"}:
        paths = build_demo_datasets(args.output_root)
        for name, path in paths.items():
            print(f"demo_dataset::{name}: {path}")


if __name__ == "__main__":
    main()
