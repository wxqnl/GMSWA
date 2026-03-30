from __future__ import annotations

import argparse
from pathlib import Path

from shiyan_benchmark.evaluation import dump_jsonl, dump_trace_csv, evaluate_experiment, load_jsonl
from shiyan_benchmark.modeling import load_lm, parse_overrides
from shiyan_benchmark.tracing import MemoryTraceRecorder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shiyan benchmark evaluation on top of lm-evaluation-harness.")
    parser.add_argument("--experiment", choices=["exp1", "exp2", "exp3", "exp4"], required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--task_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--trace_output", type=str, default=None)
    parser.add_argument("--import_module", action="append", default=[])
    parser.add_argument("--config_override", action="append", default=[])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--condition", type=str, default="default")
    parser.add_argument("--memory_source", type=str, default="none")
    parser.add_argument("--write_type", type=str, default="no_write")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enable_trace", action="store_true")
    args = parser.parse_args()

    overrides = parse_overrides(args.config_override)
    lm, model, _ = load_lm(
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

    trace_recorder = None
    if args.enable_trace:
        trace_recorder = MemoryTraceRecorder()
        trace_recorder.attach(model)

    samples = load_jsonl(args.task_file, limit=args.limit)
    rows = evaluate_experiment(
        experiment=args.experiment,
        lm=lm,
        samples=samples,
        model_name=args.model_name,
        window_size=args.window_size or overrides.get("window_size"),
        condition=args.condition,
        memory_type=args.memory_source,
        write_type=args.write_type,
        trace_recorder=trace_recorder,
    )
    dump_jsonl(args.output_file, rows)
    print(f"Wrote {len(rows)} rows to {Path(args.output_file)}")

    if trace_recorder is not None and args.trace_output:
        dump_trace_csv(args.trace_output, trace_recorder.records)
        print(f"Wrote {len(trace_recorder.records)} trace rows to {Path(args.trace_output)}")


if __name__ == "__main__":
    main()
