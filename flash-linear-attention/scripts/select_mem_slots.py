#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
FLAME_ROOT = REPO_ROOT / "flame"

STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+)\s+loss:\s*(?P<loss>[-+0-9.eE]+)\s+memory:\s*(?P<memory>[-+0-9.]+)GiB.*?tps:\s*(?P<tps>[0-9,]+)"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

PRIMARY_METRICS = {
    "train_loss": "train_loss_tail_mean",
    "eval_ppl": "eval_ppl_mean",
}


@dataclass
class StepMetric:
    step: int
    loss: float
    memory_gib: float
    tps: float


@dataclass
class TrialResult:
    num_mem_slots: int
    seed: int
    status: str
    run_dir: str
    config_path: str
    train_command: list[str]
    train_returncode: int | None = None
    train_runtime_sec: float | None = None
    train_steps_observed: int | None = None
    train_loss_tail_mean: float | None = None
    train_loss_tail_std: float | None = None
    tps_tail_mean: float | None = None
    tps_tail_std: float | None = None
    memory_gib_max: float | None = None
    memory_gib_tail_mean: float | None = None
    eval_ppl: float | None = None
    eval_tokens: int | None = None
    gate_mean: float | None = None
    gate_low_frac: float | None = None
    gate_high_frac: float | None = None
    gate_mid_frac: float | None = None
    mem_weight_mean: float | None = None
    slot_entropy: float | None = None
    effective_slot_count: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class CandidateAggregate:
    num_mem_slots: int
    seeds: list[int]
    num_trials: int
    num_successful_trials: int
    status: str
    trial_paths: list[str] = field(default_factory=list)
    quality_metric: str = ""
    quality_mean: float | None = None
    quality_std: float | None = None
    quality_se: float | None = None
    train_loss_tail_mean: float | None = None
    train_loss_tail_std: float | None = None
    train_loss_tail_se: float | None = None
    eval_ppl_mean: float | None = None
    eval_ppl_std: float | None = None
    eval_ppl_se: float | None = None
    tps_mean: float | None = None
    tps_std: float | None = None
    tps_se: float | None = None
    memory_gib_mean: float | None = None
    memory_gib_std: float | None = None
    memory_gib_se: float | None = None
    gate_mean: float | None = None
    gate_low_frac: float | None = None
    gate_high_frac: float | None = None
    gate_mid_frac: float | None = None
    mem_weight_mean: float | None = None
    slot_entropy: float | None = None
    effective_slot_count: float | None = None
    pareto_optimal: bool = False
    within_quality_band: bool = False
    balanced_eligible: bool = False
    quality_gain_vs_baseline_pct: float | None = None
    tps_change_vs_baseline_pct: float | None = None
    memory_change_vs_baseline_pct: float | None = None
    efficiency_gain_score: float | None = None
    composite_score: float | None = None
    diagnostics: list[str] = field(default_factory=list)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Sweep GM-SWA num_mem_slots and recommend paper-grade M selections."
    )
    parser.add_argument("--config", type=Path, required=True, help="Base HF config json.")
    parser.add_argument("--output-root", type=Path, required=True, help="Folder for sweep outputs.")
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=None,
        help="Candidate num_mem_slots values. Default is inferred from config.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds for repeated runs.")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Tokenizer path. If omitted, infer from forwarded train args.")
    parser.add_argument("--train-steps", type=int, default=64, help="Short training steps per candidate.")
    parser.add_argument("--tail-steps", type=int, default=8, help="Tail steps used to summarize training metrics.")
    parser.add_argument(
        "--screen-warmup-steps",
        type=int,
        default=None,
        help="Warmup steps for short training. Default uses train_steps * screen_warmup_ratio.",
    )
    parser.add_argument(
        "--screen-warmup-ratio",
        type=float,
        default=0.125,
        help="Used when screen_warmup_steps is not set.",
    )
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--master-addr", type=str, default="localhost")
    parser.add_argument("--master-port", type=str, default="0")
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--torchrun-executable", type=str, default=None)
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument(
        "--quality-metric",
        type=str,
        choices=["auto", "train_loss", "eval_ppl"],
        default="auto",
        help="Primary quality metric. 'auto' prefers eval_ppl when available.",
    )
    parser.add_argument(
        "--selection-rule",
        type=str,
        choices=["one_se", "weighted"],
        default="one_se",
        help="Balanced model selection rule. Default is one-standard-error.",
    )
    parser.add_argument("--quality-weight", type=float, default=0.6, help="Used for weighted fallback scoring.")
    parser.add_argument("--speed-weight", type=float, default=0.25, help="Used for weighted fallback scoring and cost normalization.")
    parser.add_argument("--memory-weight", type=float, default=0.15, help="Used for weighted fallback scoring and cost normalization.")
    parser.add_argument(
        "--quality-relative-tolerance",
        type=float,
        default=0.005,
        help="Relative quality tolerance for the balanced one-SE rule.",
    )
    parser.add_argument(
        "--quality-absolute-tolerance",
        type=float,
        default=0.0,
        help="Absolute quality tolerance for the balanced one-SE rule.",
    )
    parser.add_argument(
        "--min-speed-ratio",
        type=float,
        default=0.9,
        help="Balanced candidates must keep throughput above this fraction of the baseline.",
    )
    parser.add_argument(
        "--max-memory-ratio",
        type=float,
        default=1.1,
        help="Balanced candidates must keep peak memory below this multiple of the baseline.",
    )
    parser.add_argument(
        "--min-effective-slot-ratio",
        type=float,
        default=0.5,
        help="Diagnostic threshold for slot under-utilization.",
    )
    parser.add_argument(
        "--min-mem-weight",
        type=float,
        default=0.05,
        help="Diagnostic threshold for memory under-use.",
    )
    parser.add_argument(
        "--gate-saturation-threshold",
        type=float,
        default=0.8,
        help="Diagnostic threshold for heavy gate saturation.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun trials even if cached metrics exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and exit without running training.")
    parser.add_argument("--enable-wandb", action="store_true", help="Allow forwarded --metrics.enable_wandb to pass through.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--eval-dataset", type=str, default=None, help="Optional held-out evaluation dataset path/name.")
    parser.add_argument("--eval-dataset-name", type=str, default=None)
    parser.add_argument("--eval-split", type=str, default=None)
    parser.add_argument("--eval-data-dir", type=str, default=None)
    parser.add_argument("--eval-data-files", type=str, default=None, help="Comma-separated data files for load_dataset.")
    parser.add_argument("--eval-streaming", action="store_true")
    parser.add_argument("--eval-column", type=str, default="text")
    parser.add_argument("--eval-block-size", type=int, default=8192)
    parser.add_argument("--eval-max-blocks", type=int, default=8)
    parser.add_argument("--eval-max-samples", type=int, default=2048)
    parser.add_argument("--disable-diagnostics", action="store_true", help="Skip GM-SWA diagnostics during evaluation.")

    args, forwarded = parser.parse_known_args()
    if args.train_steps <= 0:
        raise ValueError("--train-steps must be > 0")
    if args.tail_steps <= 0:
        raise ValueError("--tail-steps must be > 0")
    if args.eval_max_blocks <= 0:
        raise ValueError("--eval-max-blocks must be > 0")
    if args.min_speed_ratio <= 0:
        raise ValueError("--min-speed-ratio must be > 0")
    if args.max_memory_ratio <= 0:
        raise ValueError("--max-memory-ratio must be > 0")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed")
    return args, forwarded


def split_csv(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [value.strip() for value in raw.split(",") if value.strip()]
    return values or None


def infer_default_candidates(config_dict: dict[str, Any]) -> list[int]:
    num_kv_heads = int(config_dict.get("num_kv_heads", config_dict.get("num_key_value_heads", 1)))
    if num_kv_heads <= 2:
        return [1, 2, 4]
    return [1, 2, 4, 8]


def remove_cli_options(argv: list[str], value_options: set[str], flag_options: set[str]) -> list[str]:
    cleaned: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        matched_value = next((opt for opt in value_options if arg == opt or arg.startswith(f"{opt}=")), None)
        if matched_value is not None:
            idx += 1
            if "=" not in arg and idx < len(argv):
                idx += 1
            continue
        if arg in flag_options:
            idx += 1
            continue
        cleaned.append(arg)
        idx += 1
    return cleaned


def find_option_value(argv: list[str], option: str) -> str | None:
    value: str | None = None
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == option and idx + 1 < len(argv):
            value = argv[idx + 1]
            idx += 2
            continue
        if arg.startswith(f"{option}="):
            value = arg.split("=", 1)[1]
        idx += 1
    return value


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_config(config_dict: dict[str, Any], target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(config_dict, indent=2) + "\n", encoding="utf-8")


def parse_training_metrics(log_path: Path) -> list[StepMetric]:
    metrics: list[StepMetric] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        clean_line = ANSI_RE.sub("", line)
        match = STEP_RE.search(clean_line)
        if match is None:
            continue
        metrics.append(
            StepMetric(
                step=int(match.group("step")),
                loss=float(match.group("loss")),
                memory_gib=float(match.group("memory")),
                tps=float(match.group("tps").replace(",", "")),
            )
        )
    return metrics


def summarize_training_metrics(metrics: list[StepMetric], tail_steps: int) -> dict[str, float | int]:
    if not metrics:
        return {}
    tail = metrics[-min(tail_steps, len(metrics)) :]
    losses = [metric.loss for metric in tail]
    tps_values = [metric.tps for metric in tail]
    memory_values = [metric.memory_gib for metric in tail]
    return {
        "train_steps_observed": len(metrics),
        "train_loss_tail_mean": statistics.fmean(losses),
        "train_loss_tail_std": statistics.pstdev(losses) if len(losses) > 1 else 0.0,
        "tps_tail_mean": statistics.fmean(tps_values),
        "tps_tail_std": statistics.pstdev(tps_values) if len(tps_values) > 1 else 0.0,
        "memory_gib_max": max(metric.memory_gib for metric in metrics),
        "memory_gib_tail_mean": statistics.fmean(memory_values),
    }


def build_eval_blocks(args: argparse.Namespace, tokenizer: AutoTokenizer) -> tuple[list[torch.Tensor], dict[str, Any]]:
    if args.eval_dataset is None:
        return [], {}

    dataset = load_dataset(
        path=args.eval_dataset,
        name=args.eval_dataset_name,
        split=args.eval_split or "validation",
        data_dir=args.eval_data_dir,
        data_files=split_csv(args.eval_data_files),
        streaming=args.eval_streaming,
    )

    if not args.eval_streaming and args.eval_max_samples is not None:
        dataset = dataset.select(range(min(args.eval_max_samples, len(dataset))))

    blocks: list[torch.Tensor] = []
    token_buffer: list[int] = []
    seen_samples = 0
    seen_tokens = 0
    for sample in dataset:
        if args.eval_max_samples is not None and seen_samples >= args.eval_max_samples:
            break
        text = sample.get(args.eval_column)
        if text is None:
            text = sample.get("content")
        if not text:
            continue
        input_ids = tokenizer(text, return_attention_mask=False)["input_ids"]
        if not input_ids:
            continue
        seen_samples += 1
        token_buffer.extend(input_ids)
        while len(token_buffer) >= args.eval_block_size and len(blocks) < args.eval_max_blocks:
            block = torch.tensor(token_buffer[: args.eval_block_size], dtype=torch.long)
            blocks.append(block)
            token_buffer = token_buffer[args.eval_block_size :]
            seen_tokens += int(block.numel())
        if len(blocks) >= args.eval_max_blocks:
            break

    metadata = {
        "eval_blocks": len(blocks),
        "eval_block_size": args.eval_block_size,
        "eval_seen_samples": seen_samples,
        "eval_seen_tokens": seen_tokens,
    }
    if not blocks:
        raise RuntimeError("Failed to build evaluation blocks. Check eval dataset settings.")
    return blocks, metadata


def prepare_model_imports() -> None:
    if str(FLAME_ROOT) not in sys.path:
        sys.path.insert(0, str(FLAME_ROOT))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import fla  # noqa: F401
    import custom_models  # noqa: F401


def iter_gm_swa_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [
        module
        for module in model.modules()
        if module.__class__.__name__ == "GatedMemSWA" and hasattr(module, "enable_selection_stats")
    ]


def evaluate_model(
    model_path: Path,
    eval_blocks: list[torch.Tensor],
    device: str,
    collect_diagnostics: bool,
) -> dict[str, float | int]:
    prepare_model_imports()
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device).eval()

    gm_layers = iter_gm_swa_layers(model)
    if collect_diagnostics:
        for layer in gm_layers:
            layer.enable_selection_stats(True)

    total_nll = 0.0
    total_tokens = 0
    start_time = time.perf_counter()
    with torch.inference_mode():
        for block in eval_blocks:
            input_ids = block.unsqueeze(0).to(device)
            logits = model(input_ids).logits[:, :-1].float()
            labels = input_ids[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
            total_nll += float(loss.item())
            total_tokens += int(labels.numel())

    result: dict[str, float | int] = {
        "eval_ppl": math.exp(total_nll / max(total_tokens, 1)),
        "eval_tokens": total_tokens,
        "eval_runtime_sec": time.perf_counter() - start_time,
    }
    if collect_diagnostics and gm_layers:
        stats_per_layer = [layer.get_selection_stats() for layer in gm_layers if layer.get_selection_stats()]
        if stats_per_layer:
            all_keys = sorted({key for stats in stats_per_layer for key in stats})
            for key in all_keys:
                values = [stats[key] for stats in stats_per_layer if key in stats]
                if values:
                    result[key] = statistics.fmean(values)
            if "gate_low_frac" in result and "gate_high_frac" in result:
                result["gate_mid_frac"] = max(
                    0.0,
                    1.0 - float(result["gate_low_frac"]) - float(result["gate_high_frac"]),
                )
            if "slot_entropy" in result and gm_layers[0].num_mem_slots > 1:
                result["effective_slot_count"] = float(gm_layers[0].num_mem_slots ** float(result["slot_entropy"]))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def format_float(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def stats_dict(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    se = std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "se": se,
    }


def apply_stats(target: CandidateAggregate, prefix: str, values: list[float]) -> None:
    summary = stats_dict(values)
    if summary is None:
        return
    for key, value in summary.items():
        setattr(target, f"{prefix}_{key}", value)


def prepare_env(enable_wandb: bool) -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [str(REPO_ROOT), str(FLAME_ROOT)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not enable_wandb:
        env["WANDB_MODE"] = "disabled"
    return env


def resolve_torchrun_executable(args: argparse.Namespace) -> str:
    if args.torchrun_executable is not None:
        return args.torchrun_executable
    sibling = Path(args.python_executable).with_name("torchrun")
    if sibling.exists():
        return str(sibling)
    return shutil.which("torchrun") or "torchrun"


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, dry_run: bool) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text(shlex.join(cmd) + "\n", encoding="utf-8")
        return 0, 0.0
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
        return_code = process.wait()
    return return_code, time.perf_counter() - start


def check_runtime_environment(args: argparse.Namespace, env: dict[str, str]) -> None:
    if args.skip_env_check or args.dry_run:
        return
    cmd = [
        args.python_executable,
        "-c",
        "import fla, custom_models; print('environment-ok')",
    ]
    return_code, _ = run_command(
        cmd,
        cwd=FLAME_ROOT,
        env=env,
        log_path=args.output_root / "env_check.log",
        dry_run=False,
    )
    if return_code != 0:
        raise RuntimeError(
            "Environment preflight failed. Run the sweep from the same env as successful training, "
            "or pass --python-executable/--torchrun-executable explicitly."
        )


def maybe_convert_checkpoint(
    run_dir: Path,
    config_path: Path,
    tokenizer_path: str,
    step: int,
    env: dict[str, str],
    dry_run: bool,
    python_executable: str,
) -> tuple[int, float, list[str]]:
    cmd = [
        python_executable,
        "-m",
        "flame.utils.convert_dcp_to_hf",
        "--path",
        str(run_dir),
        "--step",
        str(step),
        "--config",
        str(config_path),
        "--tokenizer",
        tokenizer_path,
    ]
    return_code, runtime_sec = run_command(
        cmd,
        cwd=FLAME_ROOT,
        env=env,
        log_path=run_dir / "convert.log",
        dry_run=dry_run,
    )
    return return_code, runtime_sec, cmd


def build_train_command(
    args: argparse.Namespace,
    forwarded_args: list[str],
    num_mem_slots: int,
    seed: int,
    candidate_config_path: Path,
    run_dir: Path,
    warmup_steps: int,
    tokenizer_path: str,
) -> list[str]:
    value_options = {
        "--job.dump_folder",
        "--job.description",
        "--model.config",
        "--model.tokenizer_path",
        "--training.steps",
        "--training.seed",
        "--lr_scheduler.warmup_steps",
        "--checkpoint.interval",
        "--checkpoint.folder",
        "--checkpoint.load_step",
        "--metrics.log_freq",
    }
    flag_options = {"--checkpoint.enable_checkpoint"}
    if not args.enable_wandb:
        flag_options.add("--metrics.enable_wandb")
    cleaned_args = remove_cli_options(forwarded_args, value_options=value_options, flag_options=flag_options)
    return [
        args.torchrun_executable,
        "--nnodes",
        str(args.nnodes),
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        f"{args.master_addr}:{args.master_port}",
        "--local-ranks-filter",
        "0",
        "-m",
        "flame.train",
        *cleaned_args,
        "--job.description",
        f"mem-slot-sweep-m{num_mem_slots}-seed{seed}",
        "--job.dump_folder",
        str(run_dir),
        "--model.config",
        str(candidate_config_path),
        "--model.tokenizer_path",
        tokenizer_path,
        "--training.seed",
        str(seed),
        "--training.steps",
        str(args.train_steps),
        "--lr_scheduler.warmup_steps",
        str(warmup_steps),
        "--checkpoint.enable_checkpoint",
        "--checkpoint.interval",
        str(args.train_steps),
        "--checkpoint.folder",
        "checkpoint",
        "--checkpoint.load_step",
        "-1",
        "--metrics.log_freq",
        "1",
    ]


def run_trial(
    args: argparse.Namespace,
    forwarded_args: list[str],
    base_config: dict[str, Any],
    num_mem_slots: int,
    seed: int,
    tokenizer_path: str,
    eval_blocks: list[torch.Tensor],
    env: dict[str, str],
) -> TrialResult:
    run_dir = args.output_root / f"m{num_mem_slots}" / f"seed{seed}"
    candidate_config_path = run_dir / f"config.m{num_mem_slots}.json"
    trial_metrics_path = run_dir / "trial_metrics.json"
    if trial_metrics_path.exists() and not args.force:
        return TrialResult(**json.loads(trial_metrics_path.read_text(encoding="utf-8")))

    config_dict = dict(base_config)
    config_dict["num_mem_slots"] = num_mem_slots
    write_config(config_dict, candidate_config_path)

    warmup_steps = args.screen_warmup_steps
    if warmup_steps is None:
        warmup_steps = max(1, int(round(args.train_steps * args.screen_warmup_ratio)))

    train_command = build_train_command(
        args=args,
        forwarded_args=forwarded_args,
        num_mem_slots=num_mem_slots,
        seed=seed,
        candidate_config_path=candidate_config_path,
        run_dir=run_dir,
        warmup_steps=warmup_steps,
        tokenizer_path=tokenizer_path,
    )
    result = TrialResult(
        num_mem_slots=num_mem_slots,
        seed=seed,
        status="running",
        run_dir=str(run_dir),
        config_path=str(candidate_config_path),
        train_command=train_command,
    )

    if args.dry_run:
        result.status = "dry_run"
        write_json(trial_metrics_path, asdict(result))
        return result

    return_code, runtime_sec = run_command(
        train_command,
        cwd=FLAME_ROOT,
        env=env,
        log_path=run_dir / "train.log",
        dry_run=False,
    )
    result.train_returncode = return_code
    result.train_runtime_sec = runtime_sec

    train_metrics = parse_training_metrics(run_dir / "train.log")
    summary = summarize_training_metrics(train_metrics, args.tail_steps)
    for key, value in summary.items():
        setattr(result, key, value)

    if return_code != 0:
        result.status = "train_failed"
        result.notes.append("training command failed")
        write_json(trial_metrics_path, asdict(result))
        return result

    if eval_blocks:
        convert_returncode, _, convert_command = maybe_convert_checkpoint(
            run_dir=run_dir,
            config_path=candidate_config_path,
            tokenizer_path=tokenizer_path,
            step=args.train_steps,
            env=env,
            dry_run=False,
            python_executable=args.python_executable,
        )
        if convert_returncode != 0:
            result.status = "convert_failed"
            result.notes.append("checkpoint conversion failed")
            result.notes.append(shlex.join(convert_command))
            write_json(trial_metrics_path, asdict(result))
            return result

        eval_metrics = evaluate_model(
            model_path=run_dir,
            eval_blocks=eval_blocks,
            device=args.device,
            collect_diagnostics=not args.disable_diagnostics,
        )
        for key, value in eval_metrics.items():
            if hasattr(result, key):
                setattr(result, key, value)

    result.status = "ok"
    write_json(trial_metrics_path, asdict(result))
    return result


def choose_quality_metric(args: argparse.Namespace, trials: list[TrialResult]) -> str:
    if args.quality_metric == "train_loss":
        return "train_loss"
    if args.quality_metric == "eval_ppl":
        return "eval_ppl"
    if any(trial.eval_ppl is not None for trial in trials):
        return "eval_ppl"
    return "train_loss"


def aggregate_trials(trials: list[TrialResult], quality_metric: str) -> list[CandidateAggregate]:
    groups: dict[int, list[TrialResult]] = defaultdict(list)
    for trial in trials:
        groups[trial.num_mem_slots].append(trial)

    aggregates: list[CandidateAggregate] = []
    quality_field = PRIMARY_METRICS[quality_metric]
    for num_mem_slots in sorted(groups):
        group = sorted(groups[num_mem_slots], key=lambda item: item.seed)
        successful = [trial for trial in group if trial.status == "ok"]
        status = "failed"
        if successful and len(successful) == len(group):
            status = "ok"
        elif successful:
            status = "partial"

        candidate = CandidateAggregate(
            num_mem_slots=num_mem_slots,
            seeds=[trial.seed for trial in group],
            num_trials=len(group),
            num_successful_trials=len(successful),
            status=status,
            trial_paths=[trial.run_dir for trial in group],
            quality_metric=quality_metric,
        )
        apply_stats(candidate, "train_loss_tail", [trial.train_loss_tail_mean for trial in successful if trial.train_loss_tail_mean is not None])
        apply_stats(candidate, "eval_ppl", [trial.eval_ppl for trial in successful if trial.eval_ppl is not None])
        apply_stats(candidate, "tps", [trial.tps_tail_mean for trial in successful if trial.tps_tail_mean is not None])
        apply_stats(candidate, "memory_gib", [trial.memory_gib_max for trial in successful if trial.memory_gib_max is not None])

        if successful:
            diag_fields = [
                "gate_mean",
                "gate_low_frac",
                "gate_high_frac",
                "gate_mid_frac",
                "mem_weight_mean",
                "slot_entropy",
                "effective_slot_count",
            ]
            for field_name in diag_fields:
                values = [getattr(trial, field_name) for trial in successful if getattr(trial, field_name) is not None]
                if values:
                    setattr(candidate, field_name, statistics.fmean(values))

        if quality_field == "train_loss_tail_mean":
            candidate.quality_mean = candidate.train_loss_tail_mean
            candidate.quality_std = candidate.train_loss_tail_std
            candidate.quality_se = candidate.train_loss_tail_se
        else:
            candidate.quality_mean = candidate.eval_ppl_mean
            candidate.quality_std = candidate.eval_ppl_std
            candidate.quality_se = candidate.eval_ppl_se

        aggregates.append(candidate)
    return aggregates


def candidate_is_valid(candidate: CandidateAggregate) -> bool:
    return (
        candidate.status == "ok"
        and candidate.quality_mean is not None
        and candidate.tps_mean is not None
        and candidate.memory_gib_mean is not None
    )


def dominates(a: CandidateAggregate, b: CandidateAggregate) -> bool:
    if not candidate_is_valid(a) or not candidate_is_valid(b):
        return False
    not_worse = (
        float(a.quality_mean) <= float(b.quality_mean)
        and float(a.tps_mean) >= float(b.tps_mean)
        and float(a.memory_gib_mean) <= float(b.memory_gib_mean)
    )
    strictly_better = (
        float(a.quality_mean) < float(b.quality_mean)
        or float(a.tps_mean) > float(b.tps_mean)
        or float(a.memory_gib_mean) < float(b.memory_gib_mean)
    )
    return not_worse and strictly_better


def compute_frontier(aggregates: list[CandidateAggregate]) -> list[int]:
    frontier: list[int] = []
    for idx, candidate in enumerate(aggregates):
        if not candidate_is_valid(candidate):
            continue
        if any(dominates(other, candidate) for jdx, other in enumerate(aggregates) if jdx != idx):
            continue
        frontier.append(idx)
    return frontier


def relative_quality_gain(baseline_quality: float, candidate_quality: float) -> float:
    return max(0.0, (baseline_quality - candidate_quality) / max(abs(baseline_quality), 1e-12))


def compute_weighted_score(
    candidate: CandidateAggregate,
    best_quality: float,
    best_tps: float,
    best_memory: float,
    quality_weight: float,
    speed_weight: float,
    memory_weight: float,
) -> float:
    quality_score = best_quality / max(float(candidate.quality_mean), 1e-12)
    speed_score = float(candidate.tps_mean) / max(best_tps, 1e-12)
    memory_score = best_memory / max(float(candidate.memory_gib_mean), 1e-12)
    return (
        quality_score ** quality_weight
        * speed_score ** speed_weight
        * memory_score ** memory_weight
    )


def enrich_aggregates(
    aggregates: list[CandidateAggregate],
    baseline: CandidateAggregate,
    args: argparse.Namespace,
) -> None:
    valid = [candidate for candidate in aggregates if candidate_is_valid(candidate)]
    if not valid:
        return
    best_quality = min(float(candidate.quality_mean) for candidate in valid)
    best_tps = max(float(candidate.tps_mean) for candidate in valid)
    best_memory = min(float(candidate.memory_gib_mean) for candidate in valid)
    cost_weight_sum = args.speed_weight + args.memory_weight
    for candidate in valid:
        candidate.quality_gain_vs_baseline_pct = 100.0 * relative_quality_gain(
            float(baseline.quality_mean),
            float(candidate.quality_mean),
        )
        candidate.tps_change_vs_baseline_pct = 100.0 * (
            float(candidate.tps_mean) / max(float(baseline.tps_mean), 1e-12) - 1.0
        )
        candidate.memory_change_vs_baseline_pct = 100.0 * (
            float(candidate.memory_gib_mean) / max(float(baseline.memory_gib_mean), 1e-12) - 1.0
        )

        speed_cost = max(0.0, 1.0 - float(candidate.tps_mean) / max(float(baseline.tps_mean), 1e-12))
        memory_cost = max(0.0, float(candidate.memory_gib_mean) / max(float(baseline.memory_gib_mean), 1e-12) - 1.0)
        normalized_cost = (
            (args.speed_weight * speed_cost + args.memory_weight * memory_cost) / max(cost_weight_sum, 1e-12)
        )
        quality_gain = relative_quality_gain(float(baseline.quality_mean), float(candidate.quality_mean))
        candidate.efficiency_gain_score = quality_gain / max(normalized_cost, 1e-6) if quality_gain > 0 else 0.0
        candidate.composite_score = compute_weighted_score(
            candidate=candidate,
            best_quality=best_quality,
            best_tps=best_tps,
            best_memory=best_memory,
            quality_weight=args.quality_weight,
            speed_weight=args.speed_weight,
            memory_weight=args.memory_weight,
        )

        if candidate.num_mem_slots > 1 and candidate.effective_slot_count is not None:
            if candidate.effective_slot_count < candidate.num_mem_slots * args.min_effective_slot_ratio:
                candidate.diagnostics.append("slots_underutilized")
        if candidate.mem_weight_mean is not None and candidate.mem_weight_mean < args.min_mem_weight:
            candidate.diagnostics.append("memory_read_weight_low")
        if candidate.gate_low_frac is not None and candidate.gate_low_frac > args.gate_saturation_threshold:
            candidate.diagnostics.append("gate_saturates_low")
        if candidate.gate_high_frac is not None and candidate.gate_high_frac > args.gate_saturation_threshold:
            candidate.diagnostics.append("gate_saturates_high")


def pick_smallest_by_quality_then_speed(candidates: list[CandidateAggregate]) -> CandidateAggregate:
    return min(
        candidates,
        key=lambda candidate: (
            candidate.num_mem_slots,
            -float(candidate.tps_mean),
            float(candidate.memory_gib_mean),
        ),
    )


def select_recommendations(
    aggregates: list[CandidateAggregate],
    args: argparse.Namespace,
) -> dict[str, Any]:
    valid = [candidate for candidate in aggregates if candidate_is_valid(candidate)]
    if not valid:
        raise RuntimeError("No valid candidates with complete metrics were found.")

    valid.sort(key=lambda candidate: candidate.num_mem_slots)
    baseline = valid[0]
    quality_optimal = min(valid, key=lambda candidate: (float(candidate.quality_mean), candidate.num_mem_slots))

    frontier_indices = compute_frontier(aggregates)
    for idx in frontier_indices:
        aggregates[idx].pareto_optimal = True

    quality_margin = max(
        float(quality_optimal.quality_se or 0.0),
        args.quality_absolute_tolerance,
        abs(float(quality_optimal.quality_mean)) * args.quality_relative_tolerance,
    )
    quality_threshold = float(quality_optimal.quality_mean) + quality_margin

    within_band = [candidate for candidate in valid if float(candidate.quality_mean) <= quality_threshold]
    for candidate in within_band:
        candidate.within_quality_band = True

    balanced_pool = [
        candidate
        for candidate in within_band
        if float(candidate.tps_mean) >= float(baseline.tps_mean) * args.min_speed_ratio
        and float(candidate.memory_gib_mean) <= float(baseline.memory_gib_mean) * args.max_memory_ratio
    ]
    for candidate in balanced_pool:
        candidate.balanced_eligible = True

    if balanced_pool:
        balanced = pick_smallest_by_quality_then_speed(balanced_pool)
    else:
        fallback_pool = [candidate for candidate in within_band if candidate.pareto_optimal]
        if fallback_pool:
            balanced = pick_smallest_by_quality_then_speed(fallback_pool)
        elif args.selection_rule == "weighted":
            balanced = max(valid, key=lambda candidate: (float(candidate.composite_score or 0.0), -candidate.num_mem_slots))
        else:
            balanced = max(
                valid,
                key=lambda candidate: (
                    float(candidate.efficiency_gain_score or 0.0),
                    -candidate.num_mem_slots,
                ),
            )

    if within_band:
        efficiency_optimal = max(
            within_band,
            key=lambda candidate: (
                float(candidate.tps_mean),
                -float(candidate.memory_gib_mean),
                -candidate.num_mem_slots,
            ),
        )
    else:
        efficiency_optimal = baseline

    return {
        "baseline_M": baseline.num_mem_slots,
        "quality_optimal_M": quality_optimal.num_mem_slots,
        "efficiency_optimal_M": efficiency_optimal.num_mem_slots,
        "balanced_M": balanced.num_mem_slots,
        "quality_threshold": quality_threshold,
        "quality_margin": quality_margin,
        "pareto_frontier": [aggregates[idx].num_mem_slots for idx in frontier_indices],
    }


def protocol_grade(args: argparse.Namespace, quality_metric: str) -> str:
    if quality_metric == "eval_ppl" and len(args.seeds) >= 2:
        return "paper_ready"
    if quality_metric == "eval_ppl":
        return "evaluation_only"
    return "screening_only"


def methodology_text(
    args: argparse.Namespace,
    candidates: list[int],
    quality_metric: str,
    recommendations: dict[str, Any],
) -> str:
    quality_name = "validation perplexity" if quality_metric == "eval_ppl" else "tail training loss"
    seed_text = ", ".join(str(seed) for seed in args.seeds)
    return (
        f"We sweep the GM-SWA memory-slot count M over {{{', '.join(map(str, candidates))}}} "
        f"using an identical training recipe and repeated runs over seeds [{seed_text}]. "
        f"For each trial, throughput and peak memory are averaged over the last {args.tail_steps} training steps, "
        f"and the primary quality metric is {quality_name}. "
        f"Candidate-level statistics are reported as means and standard errors across seeds. "
        f"We first identify the Pareto frontier over mean quality, mean throughput, and mean peak memory. "
        f"Our balanced selection follows a one-standard-error rule: among candidates whose mean {quality_name} is within "
        f"max(one standard error of the best candidate, {args.quality_relative_tolerance * 100:.2f}% relative tolerance, "
        f"and {args.quality_absolute_tolerance:.4f} absolute tolerance) of the best mean quality, "
        f"we choose the smallest M that maintains at least {args.min_speed_ratio:.2f}x the baseline throughput "
        f"and at most {args.max_memory_ratio:.2f}x the baseline peak memory. "
        f"If no candidate satisfies these constraints, we fall back to the Pareto-optimal candidate with the strongest "
        f"quality-gain-per-resource-cost trade-off."
    )


def format_aggregate_table(aggregates: list[CandidateAggregate]) -> str:
    headers = [
        "M",
        "Seeds",
        "Status",
        "Quality",
        "TPS",
        "MemGiB",
        "PPL",
        "Gate",
        "MemW",
        "EffSlots",
        "Pareto",
        "Balanced",
    ]
    rows: list[list[str]] = []
    for candidate in sorted(aggregates, key=lambda item: item.num_mem_slots):
        quality = "-"
        if candidate.quality_mean is not None:
            quality = f"{candidate.quality_mean:.5f}±{(candidate.quality_se or 0.0):.5f}"
        tps = "-"
        if candidate.tps_mean is not None:
            tps = f"{candidate.tps_mean:.0f}±{(candidate.tps_se or 0.0):.0f}"
        memory = "-"
        if candidate.memory_gib_mean is not None:
            memory = f"{candidate.memory_gib_mean:.2f}±{(candidate.memory_gib_se or 0.0):.2f}"
        ppl = "-"
        if candidate.eval_ppl_mean is not None:
            ppl = f"{candidate.eval_ppl_mean:.3f}±{(candidate.eval_ppl_se or 0.0):.3f}"
        rows.append(
            [
                str(candidate.num_mem_slots),
                str(candidate.num_successful_trials),
                candidate.status,
                quality,
                tps,
                memory,
                ppl,
                format_float(candidate.gate_mean, 3),
                format_float(candidate.mem_weight_mean, 3),
                format_float(candidate.effective_slot_count, 2),
                "yes" if candidate.pareto_optimal else "",
                "yes" if candidate.balanced_eligible else "",
            ]
        )

    widths = [max(len(header), *(len(row[idx]) for row in rows)) for idx, header in enumerate(headers)]
    sep = "  "
    lines = [sep.join(header.ljust(widths[idx]) for idx, header in enumerate(headers))]
    lines.append(sep.join("-" * width for width in widths))
    for row in rows:
        lines.append(sep.join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
    return "\n".join(lines)


def markdown_table(aggregates: list[CandidateAggregate]) -> str:
    lines = [
        "| M | Seeds | Status | Quality (mean±se) | TPS (mean±se) | MemGiB (mean±se) | Eval PPL | Pareto | Balanced | Diagnostics |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in sorted(aggregates, key=lambda item: item.num_mem_slots):
        quality = "-"
        if candidate.quality_mean is not None:
            quality = f"{candidate.quality_mean:.5f}±{(candidate.quality_se or 0.0):.5f}"
        tps = "-"
        if candidate.tps_mean is not None:
            tps = f"{candidate.tps_mean:.0f}±{(candidate.tps_se or 0.0):.0f}"
        memory = "-"
        if candidate.memory_gib_mean is not None:
            memory = f"{candidate.memory_gib_mean:.2f}±{(candidate.memory_gib_se or 0.0):.2f}"
        ppl = "-"
        if candidate.eval_ppl_mean is not None:
            ppl = f"{candidate.eval_ppl_mean:.3f}±{(candidate.eval_ppl_se or 0.0):.3f}"
        diagnostics = ", ".join(candidate.diagnostics) if candidate.diagnostics else "-"
        lines.append(
            f"| {candidate.num_mem_slots} | {candidate.num_successful_trials}/{candidate.num_trials} | {candidate.status} | "
            f"{quality} | {tps} | {memory} | {ppl} | "
            f"{'yes' if candidate.pareto_optimal else ''} | {'yes' if candidate.balanced_eligible else ''} | {diagnostics} |"
        )
    return "\n".join(lines)


def write_markdown_report(
    path: Path,
    *,
    aggregates: list[CandidateAggregate],
    recommendations: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    lines = [
        "# GM-SWA Memory-Slot Sweep",
        "",
        "## Protocol",
        "",
        protocol["methodology_text"],
        "",
        f"- Protocol grade: `{protocol['grade']}`",
        f"- Primary quality metric: `{protocol['quality_metric']}`",
        f"- Seeds: `{', '.join(map(str, protocol['seeds']))}`",
        f"- Quality-optimal M: `{recommendations['quality_optimal_M']}`",
        f"- Efficiency-optimal M: `{recommendations['efficiency_optimal_M']}`",
        f"- Balanced M: `{recommendations['balanced_M']}`",
        f"- Pareto frontier: `{', '.join(map(str, recommendations['pareto_frontier']))}`",
        "",
        "## Aggregate Results",
        "",
        markdown_table(aggregates),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args, forwarded_args = parse_args()
    args.torchrun_executable = resolve_torchrun_executable(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    base_config = load_config(args.config)
    candidates = sorted(set(args.candidates or infer_default_candidates(base_config)))
    tokenizer_path = args.tokenizer_path or find_option_value(forwarded_args, "--model.tokenizer_path")
    if tokenizer_path is None and not args.dry_run:
        raise ValueError("Tokenizer path is required. Pass --tokenizer-path or forward --model.tokenizer_path.")

    env = prepare_env(enable_wandb=args.enable_wandb)
    check_runtime_environment(args, env)

    eval_blocks: list[torch.Tensor] = []
    eval_metadata: dict[str, Any] = {}
    if args.eval_dataset is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        eval_blocks, eval_metadata = build_eval_blocks(args, tokenizer)

    trials: list[TrialResult] = []
    for num_mem_slots in candidates:
        for seed in args.seeds:
            print(f"[select_mem_slots] running M={num_mem_slots}, seed={seed}")
            trial = run_trial(
                args=args,
                forwarded_args=forwarded_args,
                base_config=base_config,
                num_mem_slots=num_mem_slots,
                seed=seed,
                tokenizer_path=tokenizer_path or "",
                eval_blocks=eval_blocks,
                env=env,
            )
            trials.append(trial)

    if args.dry_run:
        report = {
            "base_config": str(args.config),
            "output_root": str(args.output_root),
            "candidates": candidates,
            "seeds": args.seeds,
            "train_steps": args.train_steps,
            "trials": [asdict(trial) for trial in trials],
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(args.output_root / "report.json", report)
        print("Dry-run report saved to", args.output_root / "report.json")
        return

    quality_metric = choose_quality_metric(args, trials)
    aggregates = aggregate_trials(trials, quality_metric=quality_metric)
    valid_aggregates = [aggregate for aggregate in aggregates if candidate_is_valid(aggregate)]

    protocol = {
        "grade": protocol_grade(args, quality_metric),
        "quality_metric": quality_metric,
        "seeds": args.seeds,
        "train_steps": args.train_steps,
        "tail_steps": args.tail_steps,
        "eval": eval_metadata,
    }

    report: dict[str, Any] = {
        "base_config": str(args.config),
        "output_root": str(args.output_root),
        "candidates": candidates,
        "seeds": args.seeds,
        "train_steps": args.train_steps,
        "tail_steps": args.tail_steps,
        "quality_metric": quality_metric,
        "protocol": protocol,
        "trials": [asdict(trial) for trial in sorted(trials, key=lambda item: (item.num_mem_slots, item.seed))],
        "aggregates": [asdict(aggregate) for aggregate in sorted(aggregates, key=lambda item: item.num_mem_slots)],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if not valid_aggregates:
        write_json(args.output_root / "report.json", report)
        raise SystemExit(f"No valid candidates. Report saved to {args.output_root / 'report.json'}.")

    baseline = min(valid_aggregates, key=lambda aggregate: aggregate.num_mem_slots)
    enrich_aggregates(aggregates, baseline=baseline, args=args)
    recommendations = select_recommendations(aggregates, args=args)
    protocol["methodology_text"] = methodology_text(args, candidates, quality_metric, recommendations)
    report["protocol"] = protocol
    report["aggregates"] = [asdict(aggregate) for aggregate in sorted(aggregates, key=lambda item: item.num_mem_slots)]
    report["recommendations"] = recommendations
    report["paper_ready"] = protocol["grade"] == "paper_ready"
    report["notes"] = []
    if protocol["grade"] != "paper_ready":
        report["notes"].append(
            "This run is suitable for screening, but paper-grade selection should use held-out evaluation and at least two seeds."
        )

    write_json(args.output_root / "report.json", report)
    write_markdown_report(
        args.output_root / "report.md",
        aggregates=aggregates,
        recommendations=recommendations,
        protocol=protocol,
    )

    print()
    print(format_aggregate_table(aggregates))
    print()
    print(f"Quality-optimal M={recommendations['quality_optimal_M']}")
    print(f"Efficiency-optimal M={recommendations['efficiency_optimal_M']}")
    print(f"Balanced M={recommendations['balanced_M']}")
    print(f"Pareto frontier: {', '.join(map(str, recommendations['pareto_frontier']))}")
    print(f"Protocol grade: {protocol['grade']}")
    print(f"Report saved to {args.output_root / 'report.json'}")
    print(f"Markdown report saved to {args.output_root / 'report.md'}")


if __name__ == "__main__":
    main()
