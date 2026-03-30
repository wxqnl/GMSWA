# Copyright (c) 2023-2024, Songlin Yang, Yu Zhang.

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.profiler as profiler
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # noqa
from fla.layers.attn import Attention
from fla.layers.gated_mem_swa import GatedMemSWA
from fla.layers.gla import GatedLinearAttention
from fla.models.utils import Cache
from fla.modules import GatedMLP, RMSNorm


def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'


def flash_attn2_context():
    try:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_mem_efficient=False,
            enable_math=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "FlashAttention2 is not available in this PyTorch build/GPU. "
            "Please ensure torch is built with flash-attn kernels and supported GPU."
        ) from exc


def build_fair_attention(attn_kind, base_config, layer_idx, window_size, args):
    if attn_kind == "gmswa":
        return GatedMemSWA(
            dim=base_config.hidden_size,
            num_heads=base_config.num_heads,
            num_kv_heads=getattr(base_config, "num_kv_heads", None),
            qkv_bias=getattr(base_config, "qkv_bias", True),
            window_size=window_size,
            rope_theta=getattr(base_config, "rope_theta", 10000.0),
            max_position_embeddings=getattr(base_config, "max_position_embeddings", None),
            mem_scale=getattr(base_config, "mem_scale", 1.0),
            mem_rank=getattr(base_config, "mem_rank", None),
            mem_proj_mode=getattr(base_config, "mem_proj_mode", "linear"),
            mem_gate_mode=getattr(base_config, "mem_gate_mode", "linear"),
            mem_update_stride=getattr(base_config, "mem_update_stride", 1),
            mem_token_threshold=getattr(base_config, "mem_token_threshold", None),
            gate_bias_init=getattr(base_config, "gate_bias_init", 1.0),
            mem_norm=getattr(base_config, "mem_norm", True),
            mem_norm_eps=getattr(base_config, "mem_norm_eps", 1e-6),
            layer_idx=layer_idx,
        )
    if attn_kind == "swa":
        return Attention(
            hidden_size=base_config.hidden_size,
            num_heads=base_config.num_heads,
            num_kv_heads=getattr(base_config, "num_kv_heads", None),
            qkv_bias=getattr(base_config, "qkv_bias", True),
            window_size=window_size,
            rope_theta=getattr(base_config, "rope_theta", 10000.0),
            max_position_embeddings=getattr(base_config, "max_position_embeddings", None),
            layer_idx=layer_idx,
        )
    if attn_kind == "gla":
        expand_k = args.gla_expand_k if args.gla_expand_k is not None else 0.5
        expand_v = args.gla_expand_v if args.gla_expand_v is not None else 1.0
        return GatedLinearAttention(
            mode="chunk",
            hidden_size=base_config.hidden_size,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=base_config.num_heads,
            num_kv_heads=getattr(base_config, "num_kv_heads", None),
            feature_map=None,
            use_short_conv=False,
            conv_size=4,
            use_output_gate=True,
            gate_fn=getattr(base_config, "hidden_act", "swish"),
            elementwise_affine=True,
            norm_eps=getattr(base_config, "norm_eps", 1e-6),
            clamp_min=None,
            fuse_norm=getattr(base_config, "fuse_norm", True),
            layer_idx=layer_idx,
        )
    raise ValueError(f"Unknown attention kind: {attn_kind}")


class FairBlock(nn.Module):
    def __init__(self, base_config, attn_kind, layer_idx, window_size, args):
        super().__init__()
        norm_cls = RMSNorm if getattr(base_config, "fuse_norm", True) else nn.RMSNorm
        self.config = base_config
        self.attn_norm = norm_cls(base_config.hidden_size, eps=getattr(base_config, "norm_eps", 1e-6))
        self.attn = build_fair_attention(attn_kind, base_config, layer_idx, window_size, args)
        self.mlp_norm = norm_cls(base_config.hidden_size, eps=getattr(base_config, "norm_eps", 1e-6))
        self.mlp = GatedMLP(
            hidden_size=base_config.hidden_size,
            hidden_ratio=getattr(base_config, "hidden_ratio", None),
            intermediate_size=getattr(base_config, "intermediate_size", None),
            hidden_act=getattr(base_config, "hidden_act", "swish"),
            fuse_swiglu=getattr(base_config, "fuse_swiglu", True),
        )

    def forward(self, hidden_states, past_key_values=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, _, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=False,
        )
        if getattr(self.config, "fuse_norm", True):
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, past_key_values


class FairModel(nn.Module):
    def __init__(self, base_config, attn_kind, window_size, args):
        super().__init__()
        self.config = base_config
        self.vocab_size = base_config.vocab_size
        pad_token_id = getattr(base_config, "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0
        self.embeddings = nn.Embedding(base_config.vocab_size, base_config.hidden_size, pad_token_id)
        self.layers = nn.ModuleList([
            FairBlock(base_config, attn_kind, layer_idx, window_size, args)
            for layer_idx in range(base_config.num_hidden_layers)
        ])
        norm_cls = RMSNorm if getattr(base_config, "fuse_norm", True) else nn.RMSNorm
        self.norm = norm_cls(base_config.hidden_size, eps=getattr(base_config, "norm_eps", 1e-6))
        self.lm_head = nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=False)
        if getattr(base_config, "tie_word_embeddings", False):
            self.lm_head.weight = self.embeddings.weight

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        if use_cache and past_key_values is None:
            past_key_values = Cache()
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states, past_key_values = layer(
                hidden_states,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def load_config(path_or_name):
    if path_or_name is None:
        return None
    path = Path(path_or_name)
    if path.is_file():
        data = json.loads(path.read_text())
        model_type = data.get("model_type")
        if model_type is None:
            raise ValueError(f"Config file {path_or_name} is missing `model_type`.")
        data = dict(data)
        data.pop("model_type", None)
        return AutoConfig.for_model(model_type, **data)
    return AutoConfig.from_pretrained(path_or_name)


def resolve_intermediate_size(config):
    if config.intermediate_size is not None:
        return config.intermediate_size
    hidden_ratio = config.hidden_ratio if config.hidden_ratio is not None else 4
    intermediate_size = int(config.hidden_size * hidden_ratio * 2 / 3)
    return 256 * ((intermediate_size + 256 - 1) // 256)


def derive_gla_config(base_config):
    return AutoConfig.for_model(
        "gla",
        hidden_size=base_config.hidden_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_heads=base_config.num_heads,
        num_kv_heads=getattr(base_config, "num_kv_heads", None),
        hidden_ratio=getattr(base_config, "hidden_ratio", None),
        intermediate_size=getattr(base_config, "intermediate_size", None),
        hidden_act=getattr(base_config, "hidden_act", "swish"),
        max_position_embeddings=getattr(base_config, "max_position_embeddings", 2048),
        norm_eps=getattr(base_config, "norm_eps", 1e-6),
        use_cache=getattr(base_config, "use_cache", True),
        tie_word_embeddings=getattr(base_config, "tie_word_embeddings", False),
        vocab_size=getattr(base_config, "vocab_size", 32000),
        fuse_norm=getattr(base_config, "fuse_norm", True),
        fuse_swiglu=getattr(base_config, "fuse_swiglu", True),
        fuse_cross_entropy=getattr(base_config, "fuse_cross_entropy", True),
        fuse_linear_cross_entropy=getattr(base_config, "fuse_linear_cross_entropy", False),
        use_l2warp=getattr(base_config, "use_l2warp", False),
    )


def match_gla_params(target_params, gla_config):
    gla_config.intermediate_size = resolve_intermediate_size(gla_config)
    probe = AutoModelForCausalLM.from_config(gla_config)
    total_params = probe.num_parameters()
    mlp_params = 3 * gla_config.hidden_size * gla_config.intermediate_size * gla_config.num_hidden_layers
    base_params = total_params - mlp_params
    del probe

    if target_params <= base_params:
        print(
            f"Target params ({target_params}) <= non-MLP params ({base_params}); "
            "cannot match by tuning intermediate_size.",
        )
        return

    per_unit = 3 * gla_config.hidden_size * gla_config.num_hidden_layers
    desired = int(round((target_params - base_params) / per_unit))
    desired = max(256, 256 * ((desired + 256 - 1) // 256))
    gla_config.intermediate_size = desired


def build_model_from_config(config, device, dtype, use_cache):
    model = AutoModelForCausalLM.from_config(config)
    model.to(device=device, dtype=dtype)
    model.config.use_cache = use_cache
    return model


def build_model_from_pretrained(path, device, dtype, use_cache):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map={"": device},
        torch_dtype=dtype,
        use_cache=use_cache,
    )
    return model


def apply_window_size(config, window_size):
    if hasattr(config, "window_size"):
        config.window_size = window_size
        return True
    attn = getattr(config, "attn", None)
    if isinstance(attn, dict):
        attn["window_size"] = window_size
        config.attn = attn
        return True
    return False


def sample_next_token(logits, temperature, top_p, greedy=False):
    if greedy or temperature is None or temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits.float()
    logits = torch.nan_to_num(logits, neginf=-1e9, posinf=1e9)
    logits = logits / max(temperature, 1e-6)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))
        logits = torch.empty_like(sorted_logits).scatter_(-1, sorted_indices, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    prob_sum = probs.sum(dim=-1, keepdim=True)
    if (prob_sum <= 0).any():
        return torch.argmax(logits, dim=-1)
    probs = probs / prob_sum
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def run_generation_benchmark(
    label,
    model,
    tokenizer,
    input_ids,
    max_length,
    args,
    show_model=False,
):
    use_cache = not args.no_cache
    prof = None
    if args.profile:
        trace_dir = Path(args.profile_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        prof = profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            schedule=profiler.schedule(
                wait=args.profile_wait,
                warmup=args.profile_warmup,
                active=args.profile_active,
                repeat=args.profile_repeat,
            ),
            on_trace_ready=profiler.tensorboard_trace_handler(str(trace_dir / label.replace(" ", "_"))),
            record_shapes=args.profile_shapes,
            profile_memory=args.profile_memory,
            with_stack=args.profile_stack,
        )
    with flash_attn2_context():
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, use_cache=use_cache)
            next_token = sample_next_token(
                outputs.logits[:, -1, :],
                args.temperature,
                args.topp,
                greedy=args.greedy,
            )
            past_key_values = outputs.past_key_values if use_cache else None

            torch.cuda.synchronize()
            start = time.time()
            if prof is not None:
                prof.start()
            for _ in range(args.maxlen):
                if use_cache:
                    outputs = model(
                        input_ids=next_token[:, None],
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    past_key_values = outputs.past_key_values
                else:
                    input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)
                    outputs = model(input_ids=input_ids, use_cache=False)
                next_token = sample_next_token(
                    outputs.logits[:, -1, :],
                    args.temperature,
                    args.topp,
                    greedy=args.greedy,
                )
                if prof is not None:
                    prof.step()
            torch.cuda.synchronize()
            if prof is not None:
                prof.stop()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    total_tokens = input_ids.shape[0] * args.maxlen
    token_per_sec = total_tokens / max(elapsed, 1e-9)
    print(f"[{label}] decode: {token_per_sec:.2f} token/s")
    if prof is not None:
        table = prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=args.profile_rows,
        )
        print(f"[{label}] profiler top ops (self_cuda_time_total):\n{table}")
    return token_per_sec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation benchmarking")
    parser.add_argument("--path", type=str, default="/home/minko/newswa/planC/flash-linear-attention/flame/saves/qwen2-gsw-new-64")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    #parser.add_argument("--data", type=str, default="/home/minko/data/fineweb-edu")
    parser.add_argument("--prompt", type=str, default="hi!")
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--maxlen", type=int, default=256)
    parser.add_argument("--force-maxlen", action='store_true')
    parser.add_argument("--no-cache", action='store_true')
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--topp", type=float, default=0.2)
    parser.add_argument("--greedy", action='store_true')
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--output-generation", action='store_true')
    parser.add_argument("--compile", action='store_true')
    parser.add_argument("--compare-gla", action='store_true')
    parser.add_argument("--fair-attn", action='store_true')
    parser.add_argument("--gated-config", type=str, default=None)
    parser.add_argument("--gla-config", type=str, default=None)
    parser.add_argument("--qwen-config", type=str, default=None)
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--no-match-params", action='store_true')
    parser.add_argument("--gla-expand-k", type=float, default=None)
    parser.add_argument("--gla-expand-v", type=float, default=None)
    parser.add_argument("--profile", action='store_true', help="Enable torch.profiler")
    parser.add_argument("--profile-dir", type=str, default="./profiler_traces")
    parser.add_argument("--profile-wait", type=int, default=1)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-active", type=int, default=3)
    parser.add_argument("--profile-repeat", type=int, default=1)
    parser.add_argument("--profile-rows", type=int, default=30)
    parser.add_argument("--profile-shapes", action='store_true')
    parser.add_argument("--profile-memory", action='store_true')
    parser.add_argument("--profile-stack", action='store_true')
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    tokenizer_path = args.tokenizer or args.path
    print(f"Loading tokenizer {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, add_eos_token=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"{tokenizer}")

    if args.prompt_file is not None:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        print(f"Loading {args.data}")
        dataset = load_dataset(args.data, split='train')
        print(f"{dataset}")
        prompt = dataset[0]['text']
    prompts = [prompt] * max(1, args.batch_size)
    tokens = tokenizer(prompts, return_tensors="pt", padding=False)
    input_ids = tokens.input_ids.to(device=device)[:, :args.length].contiguous()
    max_length = input_ids.shape[1] + args.maxlen

    if args.fair_attn:
        if args.gated_config is None:
            raise ValueError("--gated-config is required when --fair-attn is set.")
        base_gated_config = load_config(args.gated_config)
        if base_gated_config is None:
            raise ValueError(f"Failed to load gated config from {args.gated_config}")
        if getattr(base_gated_config, "pad_token_id", None) is None:
            base_gated_config.pad_token_id = tokenizer.eos_token_id

        for window_size in args.window_sizes:
            gm_model = FairModel(base_gated_config, "gmswa", window_size, args)
            gm_model.to(device=device, dtype=dtype)
            if args.compile:
                print("Compiling the model")
                gm_model = torch.compile(gm_model)
            gm_model.eval()
            run_generation_benchmark(f"GM-SWA window={window_size}", gm_model, tokenizer, input_ids, max_length, args, show_model=False)
            del gm_model
            torch.cuda.empty_cache()

            swa_model = FairModel(base_gated_config, "swa", window_size, args)
            swa_model.to(device=device, dtype=dtype)
            if args.compile:
                print("Compiling the model")
                swa_model = torch.compile(swa_model)
            swa_model.eval()
            run_generation_benchmark(f"SWA window={window_size}", swa_model, tokenizer, input_ids, max_length, args, show_model=False)
            del swa_model
            torch.cuda.empty_cache()

        gla_model = FairModel(base_gated_config, "gla", None, args)
        gla_model.to(device=device, dtype=dtype)
        if args.compile:
            print("Compiling the model")
            gla_model = torch.compile(gla_model)
        gla_model.eval()
        run_generation_benchmark("GLA", gla_model, tokenizer, input_ids, max_length, args, show_model=False)
        del gla_model
        torch.cuda.empty_cache()
    elif args.compare_gla:
        if args.gated_config is None:
            raise ValueError("--gated-config is required when --compare-gla is set.")

        base_gated_config = load_config(args.gated_config)
        if base_gated_config is None:
            raise ValueError(f"Failed to load gated config from {args.gated_config}")

        gla_config = load_config(args.gla_config) if args.gla_config else derive_gla_config(base_gated_config)
        if args.gla_expand_k is not None:
            gla_config.expand_k = args.gla_expand_k
        if args.gla_expand_v is not None:
            gla_config.expand_v = args.gla_expand_v

        target_model = AutoModelForCausalLM.from_config(base_gated_config)
        target_params = target_model.num_parameters()
        del target_model
        if not args.no_match_params:
            match_gla_params(target_params, gla_config)

        print(f"Target params (GM-SWA): {target_params} ({sizeof_fmt(target_params)})")

        for window_size in args.window_sizes:
            gm_config = deepcopy(base_gated_config)
            gm_config.window_size = window_size
            model = build_model_from_config(gm_config, device, dtype, use_cache=not args.no_cache)
            if args.compile:
                print("Compiling the model")
                model = torch.compile(model)
            model.eval()
            label = f"GM-SWA window={window_size}"
            run_generation_benchmark(label, model, tokenizer, input_ids, max_length, args, show_model=False)
            del model
            torch.cuda.empty_cache()

        gla_model = build_model_from_config(gla_config, device, dtype, use_cache=not args.no_cache)
        if args.compile:
            print("Compiling the model")
            gla_model = torch.compile(gla_model)
        gla_model.eval()
        run_generation_benchmark("GLA", gla_model, tokenizer, input_ids, max_length, args, show_model=False)
        del gla_model
        torch.cuda.empty_cache()

        if args.qwen_config is not None:
            qwen_config = load_config(args.qwen_config)
            if qwen_config is None:
                raise ValueError(f"Failed to load qwen config from {args.qwen_config}")
            applied = False
            for window_size in args.window_sizes:
                cfg = deepcopy(qwen_config)
                applied = apply_window_size(cfg, window_size) or applied
                qwen_model = build_model_from_config(cfg, device, dtype, use_cache=not args.no_cache)
                if args.compile:
                    print("Compiling the model")
                    qwen_model = torch.compile(qwen_model)
                qwen_model.eval()
                label = f"Qwen2 window={window_size}" if applied else "Qwen2"
                run_generation_benchmark(label, qwen_model, tokenizer, input_ids, max_length, args, show_model=False)
                del qwen_model
                torch.cuda.empty_cache()
            if not applied:
                print("[Qwen2] warning: config has no window_size/attn.window_size; ran without SWA.")
    else:
        if args.config is None:
            print(f"Loading {args.path}")
            model = build_model_from_pretrained(args.path, device, dtype, use_cache=not args.no_cache)
        else:
            config = load_config(args.config)
            if config is None:
                raise ValueError(f"Failed to load config from {args.config}")
            print(f"Initializing model from {args.config}")
            model = build_model_from_config(config, device, dtype, use_cache=not args.no_cache)

        if args.compile:
            print("Compiling the model")
            model = torch.compile(model)
        model.eval()
        print(
            f"{model.config}\n{model}\nNumber of parameters: {model.num_parameters()} "
            f"({sizeof_fmt(model.num_parameters())})\n",
        )

        run_generation_benchmark(
            "Model",
            model,
            tokenizer,
            input_ids,
            max_length,
            args,
            show_model=False,
        )
