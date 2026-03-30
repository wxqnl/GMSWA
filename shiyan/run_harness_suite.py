from __future__ import annotations

import argparse
import json
from pathlib import Path

from shiyan_benchmark.harness_suite import load_suite_manifest, run_harness_suite
from shiyan_benchmark.modeling import load_lm, parse_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real benchmark suites through the local lm-evaluation-harness task registry.")
    parser.add_argument("--suite_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--import_module", action="append", default=[])
    parser.add_argument("--config_override", action="append", default=[])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--bootstrap_iters", type=int, default=0)
    parser.add_argument("--no_log_samples", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/minko/newswa/planC/shiyan/results/harness_suites",
    )
    args = parser.parse_args()

    overrides = parse_overrides(args.config_override)
    lm, _, _ = load_lm(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        import_module_names=args.import_module,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_length=args.max_length,
        trust_remote_code=True,
        config_overrides=overrides,
    )
    suite = load_suite_manifest(args.suite_file)
    result = run_harness_suite(
        suite=suite,
        lm=lm,
        output_dir=args.output_dir,
        shared_metadata={
            "pretrained": args.model_path,
            "tokenizer": args.tokenizer_path or args.model_path,
        },
        batch_size=args.batch_size,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        log_samples=not args.no_log_samples,
        continue_on_error=not args.fail_fast,
    )
    summary = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        **result,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary_csv: {Path(result['summary_path'])}")


if __name__ == "__main__":
    main()
