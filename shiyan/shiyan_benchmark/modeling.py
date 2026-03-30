from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
LM_EVAL_ROOT = REPO_ROOT / "flash-linear-attention" / "flame" / "lm-evaluation-harness"


def ensure_python_paths() -> None:
    candidates = [
        REPO_ROOT,
        REPO_ROOT / "flash-linear-attention",
        LM_EVAL_ROOT,
    ]
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def import_modules(module_names: list[str] | None) -> None:
    ensure_python_paths()
    for module_name in module_names or []:
        importlib.import_module(module_name)


def parse_dtype(dtype: str | None) -> torch.dtype | None:
    if dtype is None or dtype == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[key]


def parse_override(raw_value: str) -> Any:
    lower = raw_value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null" or lower == "none":
        return None
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        pass
    if raw_value.startswith("{") or raw_value.startswith("["):
        return json.loads(raw_value)
    return raw_value


def parse_overrides(entries: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Invalid override '{entry}'. Expected key=value.")
        key, value = entry.split("=", 1)
        overrides[key] = parse_override(value)
    return overrides


def build_model(
    model_path: str,
    *,
    tokenizer_path: str | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    config_overrides: dict[str, Any] | None = None,
) -> tuple[torch.nn.Module, Any]:
    ensure_python_paths()
    config_overrides = dict(config_overrides or {})
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        **config_overrides,
    )
    torch_dtype = parse_dtype(dtype)
    load_kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or model_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.to(device)
    return model, tokenizer


def load_lm(
    *,
    model_path: str,
    tokenizer_path: str | None = None,
    import_module_names: list[str] | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    batch_size: int = 1,
    max_length: int | None = None,
    trust_remote_code: bool = True,
    config_overrides: dict[str, Any] | None = None,
):
    ensure_python_paths()
    import_modules(import_module_names)
    model, tokenizer = build_model(
        model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        config_overrides=config_overrides,
    )
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    return lm, model, tokenizer
