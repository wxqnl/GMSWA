#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import types
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
FLA = ROOT / "flash-linear-attention"
if str(FLA) not in sys.path:
    sys.path.insert(0, str(FLA))

import fla  # noqa: F401,E402
from fla.layers.gated_mem_swa import GatedMemSWA  # noqa: E402
from flame.data import build_dataloader, build_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GM-SWA v2 memory-on/off ablation.")
    parser.add_argument("--model-path", default=str(FLA / "flame/saves/gated_mem_swa-340M"))
    parser.add_argument("--dataset", default="/mnt/data/wuwei/data/fineweb-edu-100BT-parquet-sharded")
    parser.add_argument("--output", default=str(ROOT / "eval_results/gated_mem_swa-340M/v2_ablation.json"))
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def set_memory_enabled(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, GatedMemSWA):
            module.memory_enabled = enabled and not module.disable_memory


def iter_gate_modules(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, GatedMemSWA) and module.gate_proj is not None:
            yield name, module


def make_gate_accumulator(model: torch.nn.Module):
    stats = defaultdict(lambda: defaultdict(float))
    handles = []
    control = {"enabled": True}

    def hook_for(name: str, module: GatedMemSWA):
        def hook(_module, _inputs, output):
            if not control["enabled"]:
                return
            beta_logits, _a_logits, mix_logits = output.detach().float().split(module.num_heads, dim=-1)
            beta = torch.sigmoid(beta_logits)
            alpha = torch.sigmoid(mix_logits)
            count = float(alpha.numel())
            stats[name]["count"] += count
            stats[name]["alpha_sum"] += float(alpha.sum().item())
            stats[name]["mem_weight_sum"] += float((1.0 - alpha).sum().item())
            stats[name]["beta_sum"] += float(beta.sum().item())
            stats[name]["alpha_lt_095"] += float((alpha < 0.95).sum().item())
            stats[name]["alpha_lt_098"] += float((alpha < 0.98).sum().item())

        return hook

    for name, module in iter_gate_modules(model):
        handles.append(module.gate_proj.register_forward_hook(hook_for(name, module)))
    return stats, handles, control


@contextlib.contextmanager
def force_memory_zero(model: torch.nn.Module):
    originals = []

    def zero_memory_branch(
        self,
        q,
        k,
        v,
        hidden_states,
        *,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    ):
        final_state = None
        if output_final_state:
            batch_size = q.shape[0] if cu_seqlens is None else int(cu_seqlens.numel() - 1)
            final_state = self._new_memory_state(batch_size, q.device)
        return torch.zeros_like(q), final_state

    for module in model.modules():
        if isinstance(module, GatedMemSWA):
            originals.append((module, module._memory_branch))
            module._memory_branch = types.MethodType(zero_memory_branch, module)
    try:
        yield
    finally:
        for module, original in originals:
            module._memory_branch = original


def init_loss_bins() -> dict[str, dict[str, float]]:
    return {
        "pos_0_511": {"sum_on": 0.0, "sum_zero": 0.0, "sum_local": 0.0, "count": 0.0},
        "pos_512_1023": {"sum_on": 0.0, "sum_zero": 0.0, "sum_local": 0.0, "count": 0.0},
        "pos_1024_2047": {"sum_on": 0.0, "sum_zero": 0.0, "sum_local": 0.0, "count": 0.0},
        "pos_2048_4095": {"sum_on": 0.0, "sum_zero": 0.0, "sum_local": 0.0, "count": 0.0},
        "pos_4096_plus": {"sum_on": 0.0, "sum_zero": 0.0, "sum_local": 0.0, "count": 0.0},
    }


def bin_name(pos: int) -> str:
    if pos < 512:
        return "pos_0_511"
    if pos < 1024:
        return "pos_512_1023"
    if pos < 2048:
        return "pos_1024_2047"
    if pos < 4096:
        return "pos_2048_4095"
    return "pos_4096_plus"


def valid_positions_from_cu(cu_seqlens: torch.Tensor, total_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    cu = cu_seqlens.flatten().to("cpu", dtype=torch.long).tolist()
    valid = torch.zeros(total_len - 1, dtype=torch.bool)
    relpos = torch.zeros(total_len - 1, dtype=torch.long)
    for start, end in zip(cu[:-1], cu[1:], strict=False):
        if end - start < 2:
            continue
        lo, hi = start, end - 1
        valid[lo:hi] = True
        relpos[lo:hi] = torch.arange(hi - lo, dtype=torch.long)
    return valid, relpos


def token_losses(model, input_ids, cu_seqlens) -> torch.Tensor:
    output = model(input_ids=input_ids, cu_seqlens=cu_seqlens, use_cache=False, logits_to_keep=0)
    logits = output.logits[:, :-1].float()
    targets = input_ids[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view(-1)


def summarize_bins(bins: dict[str, dict[str, float]]) -> dict[str, dict[str, float | int | None]]:
    out = {}
    for name, values in bins.items():
        count = int(values["count"])
        if count == 0:
            out[name] = {
                "count": 0,
                "nll_on": None,
                "nll_memory_zero": None,
                "delta_zero_minus_on": None,
                "nll_local_only": None,
                "delta_local_minus_on": None,
            }
            continue
        nll_on = values["sum_on"] / count
        nll_zero = values["sum_zero"] / count
        nll_local = values["sum_local"] / count
        out[name] = {
            "count": count,
            "nll_on": nll_on,
            "ppl_on": math.exp(min(nll_on, 20.0)),
            "nll_memory_zero": nll_zero,
            "ppl_memory_zero": math.exp(min(nll_zero, 20.0)),
            "delta_zero_minus_on": nll_zero - nll_on,
            "nll_local_only": nll_local,
            "ppl_local_only": math.exp(min(nll_local, 20.0)),
            "delta_local_minus_on": nll_local - nll_on,
        }
    return out


def summarize_gate_stats(stats) -> dict[str, dict[str, float]]:
    summary = {}
    for name, values in stats.items():
        count = max(values["count"], 1.0)
        summary[name] = {
            "alpha_mean": values["alpha_sum"] / count,
            "memory_weight_mean": values["mem_weight_sum"] / count,
            "beta_mean": values["beta_sum"] / count,
            "frac_alpha_lt_0.95": values["alpha_lt_095"] / count,
            "frac_alpha_lt_0.98": values["alpha_lt_098"] / count,
        }
    if summary:
        keys = next(iter(summary.values())).keys()
        summary["_mean_over_layers"] = {
            key: sum(layer[key] for layer in summary.values()) / len(summary)
            for key in keys
        }
    return summary


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    dataset = build_dataset(
        dataset=args.dataset,
        dataset_split="train",
        streaming=True,
        num_workers=0,
        seed=args.seed,
    )
    dataloader = build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        rank=0,
        world_size=1,
        batch_size=1,
        seq_len=args.seq_len,
        context_len=args.context_len,
        varlen=True,
        num_workers=0,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    gate_stats, handles, gate_control = make_gate_accumulator(model)
    bins = init_loss_bins()
    total_on = total_zero = total_local = total_count = 0.0

    iterator = iter(dataloader)
    for batch_idx in range(args.num_batches):
        batch = next(iterator)
        input_ids = batch["input_ids"].to(args.device)
        cu_seqlens = batch["cu_seqlens"].to(args.device)
        valid, relpos = valid_positions_from_cu(batch["cu_seqlens"], input_ids.shape[1])

        set_memory_enabled(model, True)
        gate_control["enabled"] = True
        losses_on = token_losses(model, input_ids, cu_seqlens).cpu()
        gate_control["enabled"] = False
        with force_memory_zero(model):
            losses_zero = token_losses(model, input_ids, cu_seqlens).cpu()
        set_memory_enabled(model, False)
        losses_local = token_losses(model, input_ids, cu_seqlens).cpu()
        set_memory_enabled(model, True)

        valid_losses_on = losses_on[valid]
        valid_losses_zero = losses_zero[valid]
        valid_losses_local = losses_local[valid]
        total_on += float(valid_losses_on.sum().item())
        total_zero += float(valid_losses_zero.sum().item())
        total_local += float(valid_losses_local.sum().item())
        total_count += float(valid_losses_on.numel())

        valid_relpos = relpos[valid]
        for idx, pos in enumerate(valid_relpos.tolist()):
            name = bin_name(pos)
            bins[name]["sum_on"] += float(valid_losses_on[idx].item())
            bins[name]["sum_zero"] += float(valid_losses_zero[idx].item())
            bins[name]["sum_local"] += float(valid_losses_local[idx].item())
            bins[name]["count"] += 1.0

        print(
            f"batch {batch_idx + 1}/{args.num_batches}: "
            f"nll_on={valid_losses_on.mean().item():.4f} "
            f"nll_zero={valid_losses_zero.mean().item():.4f} "
            f"delta_zero={valid_losses_zero.mean().item() - valid_losses_on.mean().item():+.4f} "
            f"nll_local={valid_losses_local.mean().item():.4f} "
            f"delta_local={valid_losses_local.mean().item() - valid_losses_on.mean().item():+.4f}",
            flush=True,
        )

    for handle in handles:
        handle.remove()
    set_memory_enabled(model, True)

    overall_on = total_on / total_count
    overall_zero = total_zero / total_count
    overall_local = total_local / total_count
    result = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "seq_len": args.seq_len,
        "context_len": args.context_len,
        "num_batches": args.num_batches,
        "tokens_evaluated": int(total_count),
        "overall": {
            "nll_on": overall_on,
            "ppl_on": math.exp(min(overall_on, 20.0)),
            "nll_memory_zero": overall_zero,
            "ppl_memory_zero": math.exp(min(overall_zero, 20.0)),
            "delta_zero_minus_on": overall_zero - overall_on,
            "nll_local_only": overall_local,
            "ppl_local_only": math.exp(min(overall_local, 20.0)),
            "delta_local_minus_on": overall_local - overall_on,
        },
        "bins": summarize_bins(bins),
        "gate_stats": summarize_gate_stats(gate_stats),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["overall"], indent=2), flush=True)
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
