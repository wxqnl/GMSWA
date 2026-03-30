# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import os
import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via "
        "`pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None
    flash_attn_varlen_func = None

from fla.layers.utils import unpad_input
from fla.modules import RotaryEmbedding
from fla.ops.gm_swa import fused_recurrent_gm_swa
from fla.ops.utils.index import prepare_lens, prepare_lens_from_mask, prepare_position_ids, prepare_sequence_ids

if TYPE_CHECKING:
    from fla.models.utils import Cache


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, seq_len, num_kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, seq_len, n_rep, num_kv_heads, head_dim)
    return hidden_states.reshape(batch, seq_len, num_kv_heads * n_rep, head_dim)


class GatedMemSWA(nn.Module):
    """
    Gated Memory-Augmented Sliding Window Attention (GM-SWA).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        num_kv_heads: int | None = None,
        qkv_bias: bool = True,
        rope_theta: float = 10000.0,
        max_position_embeddings: int | None = None,
        num_mem_slots: int = 1,
        mem_scale: float = 1.0,
        mem_rank: int | None = None,
        mem_proj_mode: str = "linear",
        mem_gate_mode: str = "linear",
        mem_update_source: str = "kv",
        mem_update_stride: int = 1,
        mem_token_threshold: int | None = None,
        disable_memory: bool = False,
        gate_bias_init: float = 1.0,
        mem_norm: bool = True,
        mem_norm_eps: float = 1e-6,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}")
        if window_size is None:
            raise ValueError("window_size must be set for GatedMemSWA")
        if num_mem_slots < 0:
            raise ValueError("num_mem_slots must be >= 0")

        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.window_size = window_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.num_mem_slots = num_mem_slots
        self.gate_num_slots = 1 if num_mem_slots > 1 else num_mem_slots
        self.mem_rank = mem_rank
        self.mem_proj_mode = mem_proj_mode
        self.mem_gate_mode = mem_gate_mode
        self.mem_update_source = mem_update_source
        self.mem_update_stride = mem_update_stride
        self.mem_token_threshold = mem_token_threshold
        self.disable_memory = disable_memory
        self.memory_enabled = (not disable_memory) and num_mem_slots > 0
        self.gate_bias_init = float(gate_bias_init)
        self.gate_clamp_eps = 1e-4
        self.multi_slot_log_prior = math.log(self.num_mem_slots) if self.num_mem_slots > 1 else 0.0
        self.mem_scale_init = float(mem_scale)

        mem_scale = float(mem_scale)
        if mem_scale <= 0:
            raise ValueError("mem_scale must be > 0 when using learnable scale.")
        if mem_proj_mode not in {"linear", "scale"}:
            raise ValueError(f"Unsupported mem_proj_mode: {mem_proj_mode}")
        if mem_gate_mode not in {"linear", "param"}:
            raise ValueError(f"Unsupported mem_gate_mode: {mem_gate_mode}")
        if mem_update_source not in {"value", "kv"}:
            raise ValueError(f"Unsupported mem_update_source: {mem_update_source}")
        if mem_update_stride <= 0:
            raise ValueError("mem_update_stride must be >= 1")
        if mem_rank is not None and mem_rank <= 0:
            raise ValueError("mem_rank must be > 0 when provided.")

        self.log_mem_scale = nn.Parameter(torch.log(torch.tensor([mem_scale])))
        self.mem_norm = mem_norm
        self.mem_norm_eps = mem_norm_eps

        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)

        if self.memory_enabled and self.mem_gate_mode == "linear":
            self.gate_net = nn.Linear(dim, self.num_kv_heads * self.gate_num_slots, bias=True)
            nn.init.constant_(self.gate_net.bias, gate_bias_init)
            self.gate_param = None
        elif self.memory_enabled:
            self.gate_net = None
            self.gate_param = nn.Parameter(
                torch.full((self.num_kv_heads, self.gate_num_slots), float(gate_bias_init))
            )
        else:
            self.gate_net = None
            self.gate_param = None

        if self.memory_enabled and self.mem_proj_mode == "linear":
            if self.mem_rank is None:
                self.mem_proj = nn.Linear(self.head_dim, self.head_dim * self.num_mem_slots, bias=False)
                self.mem_proj_in = None
                self.mem_proj_out = None
            else:
                self.mem_proj = None
                self.mem_proj_in = nn.Linear(self.head_dim, self.mem_rank, bias=False)
                self.mem_proj_out = nn.Linear(self.mem_rank, self.head_dim * self.num_mem_slots, bias=False)
            self.mem_proj_scale = None
        elif self.memory_enabled:
            self.mem_proj = None
            self.mem_proj_in = None
            self.mem_proj_out = None
            self.mem_proj_scale = nn.Parameter(torch.ones(self.num_kv_heads, self.num_mem_slots, self.head_dim))
        else:
            self.mem_proj = None
            self.mem_proj_in = None
            self.mem_proj_out = None
            self.mem_proj_scale = None

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=rope_theta)
        self._selection_stats_enabled = False
        self._selection_metric_sums: dict[str, float] = {}
        self._selection_metric_counts: dict[str, float] = {}
        self._debug_nonfinite_enabled = os.getenv("GMSWA_DEBUG_NONFINITE", "0") == "1"
        self._debug_large_threshold = float(os.getenv("GMSWA_DEBUG_LARGE_THRESHOLD", "1e3"))
        self._debug_value_checks_enabled = os.getenv("GMSWA_DEBUG_VALUE_CHECKS", "1") == "1"
        self._debug_param_hooks_enabled = os.getenv("GMSWA_DEBUG_PARAM_HOOKS", "1") == "1"
        self._debug_tensor_grad_hooks_enabled = os.getenv("GMSWA_DEBUG_TENSOR_GRAD_HOOKS", "1") == "1"
        debug_patterns = os.getenv("GMSWA_DEBUG_VALUE_NAMES", "")
        self._debug_value_name_filters = tuple(p.strip() for p in debug_patterns.split(",") if p.strip())
        self._debug_param_norm_threshold = float(os.getenv("GMSWA_DEBUG_PARAM_NORM_THRESHOLD", "0"))
        self._init_memory_parameters()
        self._register_debug_parameter_hooks()

    def reset_parameters(self) -> None:
        self._init_memory_parameters()

    def _register_debug_parameter_hooks(self) -> None:
        if not self._debug_nonfinite_enabled or not self._debug_param_hooks_enabled:
            return

        for name, param in self.named_parameters():
            def _hook(grad: torch.Tensor, *, _name: str = name) -> torch.Tensor:
                self._debug_nonfinite(f"param:{_name}.grad", grad)
                self._debug_large(f"param:{_name}.grad", grad)
                self._debug_param_grad_norm(f"param:{_name}.grad", grad)
                return grad

            param.register_hook(_hook)

    def _init_memory_parameters(self) -> None:
        if not self.memory_enabled:
            return
        with torch.no_grad():
            self.log_mem_scale.fill_(math.log(self.mem_scale_init))
            if self.mem_proj_scale is not None:
                self.mem_proj_scale.fill_(1.0)
        if self.gate_net is not None:
            nn.init.zeros_(self.gate_net.weight)
            nn.init.constant_(self.gate_net.bias, self.gate_bias_init)
        elif self.gate_param is not None:
            nn.init.constant_(self.gate_param, self.gate_bias_init)
        proj_gain = 1.0 / math.sqrt(max(self.num_mem_slots, 1))
        if self.mem_proj is not None:
            nn.init.xavier_uniform_(self.mem_proj.weight, gain=proj_gain)
        if self.mem_proj_in is not None:
            nn.init.xavier_uniform_(self.mem_proj_in.weight)
        if self.mem_proj_out is not None:
            nn.init.xavier_uniform_(self.mem_proj_out.weight, gain=proj_gain)

    def enable_selection_stats(self, enabled: bool = True) -> None:
        self._selection_stats_enabled = enabled
        if enabled:
            self.reset_selection_stats()

    def reset_selection_stats(self) -> None:
        self._selection_metric_sums = {}
        self._selection_metric_counts = {}

    def get_selection_stats(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        for name, total in self._selection_metric_sums.items():
            count = self._selection_metric_counts.get(name, 0.0)
            if count > 0:
                stats[name] = total / count
        return stats

    def _accumulate_selection_metric(
        self,
        name: str,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        if not self._selection_stats_enabled:
            return
        with torch.no_grad():
            metric_values = values.detach().float()
            if mask is not None:
                metric_values = metric_values.masked_select(mask.detach())
            else:
                metric_values = metric_values.reshape(-1)
            if metric_values.numel() == 0:
                return
            self._selection_metric_sums[name] = self._selection_metric_sums.get(name, 0.0) + float(metric_values.sum().item())
            self._selection_metric_counts[name] = self._selection_metric_counts.get(name, 0.0) + float(metric_values.numel())

    def _debug_nonfinite(self, name: str, tensor: torch.Tensor) -> None:
        if (
            not self._debug_nonfinite_enabled
            or not self._debug_value_checks_enabled
            or not self._debug_track_value_name(name)
            or not torch.is_tensor(tensor)
        ):
            return
        if torch.isfinite(tensor).all():
            return
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        with torch.no_grad():
            finite_mask = torch.isfinite(tensor)
            finite_values = tensor.detach().float().masked_select(finite_mask)
            finite_amax = float(finite_values.abs().max().item()) if finite_values.numel() > 0 else float("nan")
            print(
                f"[gmswa-debug][rank {rank}][layer {self.layer_idx}] nonfinite {name}: "
                f"shape={tuple(tensor.shape)} finite_amax={finite_amax}",
                flush=True,
            )

    def _debug_large(self, name: str, tensor: torch.Tensor) -> None:
        if (
            not self._debug_nonfinite_enabled
            or not self._debug_value_checks_enabled
            or not self._debug_track_value_name(name)
            or not torch.is_tensor(tensor)
        ):
            return
        with torch.no_grad():
            finite_mask = torch.isfinite(tensor)
            finite_values = tensor.detach().float().masked_select(finite_mask)
            if finite_values.numel() == 0:
                return
            finite_amax = float(finite_values.abs().max().item())
            if finite_amax <= self._debug_large_threshold:
                return
            rank = 0
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            print(
                f"[gmswa-debug][rank {rank}][layer {self.layer_idx}] large {name}: "
                f"shape={tuple(tensor.shape)} finite_amax={finite_amax}",
                flush=True,
            )

    def _debug_nonfinite_grad(self, name: str, tensor: torch.Tensor) -> None:
        if (
            not self._debug_nonfinite_enabled
            or not self._debug_tensor_grad_hooks_enabled
            or not torch.is_tensor(tensor)
            or not tensor.requires_grad
        ):
            return

        def _hook(grad: torch.Tensor) -> torch.Tensor:
            self._debug_nonfinite(f"{name}.grad", grad)
            self._debug_large(f"{name}.grad", grad)
            self._debug_param_grad_norm(f"{name}.grad", grad)
            return grad

        tensor.register_hook(_hook)

    def _debug_track_value_name(self, name: str) -> bool:
        if not self._debug_value_name_filters:
            return True
        return any(pattern in name for pattern in self._debug_value_name_filters)

    def _debug_param_grad_norm(self, name: str, tensor: torch.Tensor) -> None:
        if (
            not self._debug_nonfinite_enabled
            or not self._debug_value_checks_enabled
            or not self._debug_track_value_name(name)
            or self._debug_param_norm_threshold <= 0
            or not torch.is_tensor(tensor)
        ):
            return
        with torch.no_grad():
            grad = tensor.detach().float()
            if not torch.isfinite(grad).all():
                return
            grad_norm = float(grad.norm().item())
            if grad_norm <= self._debug_param_norm_threshold:
                return
            grad_amax = float(grad.abs().max().item())
            rank = 0
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            print(
                f"[gmswa-debug][rank {rank}][layer {self.layer_idx}] grad-norm {name}: "
                f"norm={grad_norm} amax={grad_amax} shape={tuple(tensor.shape)}",
                flush=True,
            )

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        seqlen_offset: int | torch.Tensor = 0,
        max_seqlen: int | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_seqlen is None:
            if isinstance(seqlen_offset, int):
                max_seqlen = seqlen_offset + q.shape[1]
            else:
                max_seqlen = q.shape[1] + int(seqlen_offset.max().item())
        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        return self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

    def _normalize_memory(self, memory_state: torch.Tensor) -> torch.Tensor:
        if not self.mem_norm:
            return memory_state
        # Keep the memory bounded in forward without backpropagating through the norm.
        scale = memory_state.detach().float().square().sum(dim=-1, keepdim=True).clamp_min(1.0).sqrt_()
        return memory_state / scale.to(dtype=memory_state.dtype)

    def _memory_kv(self, memory_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mem_scale = self.mem_scale.to(device=memory_state.device, dtype=memory_state.dtype)
        return memory_state, memory_state * mem_scale

    def _expand_kv_to_query_heads(self, kv_states: torch.Tensor) -> torch.Tensor:
        if self.num_kv_groups == 1:
            return kv_states
        if kv_states.dim() == 3:
            return kv_states.repeat_interleave(self.num_kv_groups, dim=1)
        if kv_states.dim() == 4:
            return kv_states.repeat_interleave(self.num_kv_groups, dim=1)
        raise ValueError(f"Unsupported kv state rank: {kv_states.dim()}")

    def _collapse_query_groups(self, query_states: torch.Tensor) -> torch.Tensor:
        if self.num_kv_groups == 1:
            return query_states
        if query_states.dim() == 3:
            batch_size, _, head_dim = query_states.shape
            return query_states.view(batch_size, self.num_kv_heads, self.num_kv_groups, head_dim)[:, :, 0]
        if query_states.dim() == 4:
            batch_size, seq_len, _, head_dim = query_states.shape
            return query_states.view(batch_size, seq_len, self.num_kv_heads, self.num_kv_groups, head_dim)[:, :, :, 0]
        raise ValueError(f"Unsupported query state rank: {query_states.dim()}")

    def _memory_kv_for_queries(self, memory_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mem_k, mem_v = self._memory_kv(memory_state)
        return self._expand_kv_to_query_heads(mem_k), self._expand_kv_to_query_heads(mem_v)

    @property
    def mem_scale(self) -> torch.Tensor:
        return torch.exp(self.log_mem_scale).clamp_min(self.mem_norm_eps)

    def _compute_gate(self, gate_input: torch.Tensor) -> torch.Tensor:
        if not self.memory_enabled:
            raise RuntimeError("memory gate requested while memory is disabled")
        if self.mem_gate_mode == "linear":
            gate = self.gate_net(gate_input)
            gate = gate.unflatten(-1, (self.num_kv_heads, self.gate_num_slots))
            gate = torch.sigmoid(gate.float()).to(dtype=gate.dtype)
        else:
            gate = torch.sigmoid(self.gate_param.float()).to(dtype=gate_input.dtype)
            if gate_input.dim() == 3:
                gate = gate.view(1, 1, self.num_kv_heads, self.gate_num_slots).expand(
                    gate_input.shape[0], gate_input.shape[1], -1, -1
                )
            else:
                gate = gate.view(1, self.num_kv_heads, self.gate_num_slots).expand(gate_input.shape[0], -1, -1)
        return gate.clamp(min=self.gate_clamp_eps, max=1.0 - self.gate_clamp_eps)

    def _memory_update_input(self, evicted_k: torch.Tensor | None, evicted_v: torch.Tensor) -> torch.Tensor:
        if self.mem_update_source == "value":
            return evicted_v
        if evicted_k is None:
            raise RuntimeError("memory update source 'kv' requires evicted keys")
        # Keys are score-space features while values are content-space features.
        # Reusing the attention scaling keeps the key branch numerically aligned
        # with values in packed long-sequence training, without removing its
        # gradient contribution to the memory update.
        return (evicted_k * self.scaling + evicted_v) * math.sqrt(0.5)

    def _project_memory_update(self, evicted_k: torch.Tensor | None, evicted_v: torch.Tensor) -> torch.Tensor:
        if not self.memory_enabled:
            raise RuntimeError("memory projection requested while memory is disabled")
        update_input = self._memory_update_input(evicted_k, evicted_v)
        self._debug_large("update_input", update_input)
        source_scale = update_input.detach().float().norm(dim=-1, keepdim=True).clamp_min(1.0)
        update_input = update_input / source_scale.to(dtype=update_input.dtype)
        self._debug_large("update_input_scaled", update_input)
        if self.mem_proj_mode == "scale":
            scale = self.mem_proj_scale.to(device=update_input.device, dtype=update_input.dtype)
            if update_input.dim() == 3:
                projected = update_input.unsqueeze(-2) * scale.unsqueeze(0)
                self._debug_large("projected_update", projected)
                return self._normalize_memory(projected.to(torch.float32)).to(dtype=update_input.dtype)
            if update_input.dim() == 4:
                projected = update_input.unsqueeze(-2) * scale.view(
                    1,
                    1,
                    self.num_kv_heads,
                    self.num_mem_slots,
                    self.head_dim,
                )
                self._debug_large("projected_update", projected)
                return self._normalize_memory(projected.to(torch.float32)).to(dtype=update_input.dtype)
            raise ValueError(f"Unsupported memory update rank: {update_input.dim()}")

        if self.mem_proj is not None:
            projected = self.mem_proj(update_input)
        else:
            projected = self.mem_proj_out(self.mem_proj_in(update_input))
        self._debug_large("projected_update", projected)
        projected = projected.unflatten(-1, (self.num_mem_slots, self.head_dim))
        return self._normalize_memory(projected.to(torch.float32)).to(dtype=update_input.dtype)

    def _should_update_memory(self, seen_tokens_before_append: int) -> bool:
        if seen_tokens_before_append < self.window_size:
            return False
        evicted_idx = seen_tokens_before_append - self.window_size
        return evicted_idx % self.mem_update_stride == 0

    def _should_use_memory(self, seen_tokens_before_append: int) -> bool:
        if not self.memory_enabled:
            return False
        if seen_tokens_before_append < self.window_size:
            return False
        if self.mem_token_threshold is not None and (seen_tokens_before_append + 1) < self.mem_token_threshold:
            return False
        return True

    def _update_memory(
        self,
        memory_state: torch.Tensor | None,
        evicted_k: torch.Tensor | None,
        evicted_v: torch.Tensor,
        gate_input: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.memory_enabled or memory_state is None:
            return memory_state
        gate = self._compute_gate(gate_input).unsqueeze(-1).to(device=memory_state.device, dtype=memory_state.dtype)
        mem_update = self._project_memory_update(evicted_k, evicted_v).to(
            device=memory_state.device,
            dtype=memory_state.dtype,
        )
        return gate * memory_state + (1.0 - gate) * mem_update

    def _load_cached_state(
        self,
        past_key_values: Any | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        memory_state = None
        if self.memory_enabled:
            memory_state = torch.zeros(
                (batch_size, self.num_kv_heads, self.num_mem_slots, self.head_dim),
                device=device,
                dtype=dtype,
            )
        k_cached = None
        v_cached = None
        k_write_cached = None

        if past_key_values is None or self.layer_idx is None:
            return memory_state, k_cached, v_cached, k_write_cached
        if len(past_key_values) <= self.layer_idx:
            return memory_state, k_cached, v_cached, k_write_cached

        state = past_key_values[self.layer_idx]
        if state is None:
            return memory_state, k_cached, v_cached, k_write_cached

        cached_mem = state.get("recurrent_state")
        if cached_mem is not None and self.memory_enabled:
            if cached_mem.dim() == 3:
                if cached_mem.shape[1] == self.num_heads:
                    cached_mem = self._collapse_query_groups(cached_mem)
                cached_mem = cached_mem.unsqueeze(-2)
            memory_state = cached_mem.to(device=device, dtype=dtype)

        attn_state = state.get("attn_state")
        if attn_state is not None:
            if len(attn_state) == 3:
                k_flat, v_flat, k_write_flat = attn_state
                k_write_cached = k_write_flat.view(batch_size, -1, self.num_kv_heads, self.head_dim).to(
                    device=device,
                    dtype=dtype,
                )
            else:
                k_flat, v_flat = attn_state
            k_cached = k_flat.view(batch_size, -1, self.num_heads, self.head_dim).to(device=device, dtype=dtype)
            v_cached = v_flat.view(batch_size, -1, self.num_heads, self.head_dim).to(device=device, dtype=dtype)
            if k_write_cached is None:
                k_write_cached = self._collapse_query_groups(k_cached)

        return memory_state, k_cached, v_cached, k_write_cached

    def _prepare_cache_state(
        self,
        past_key_values: Any | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        memory_state, k_cached, v_cached, k_write_cached = self._load_cached_state(
            past_key_values,
            batch_size,
            device,
            dtype,
        )
        window_k = [] if k_cached is None else [k_cached[:, i] for i in range(k_cached.shape[1])]
        window_v = [] if v_cached is None else [v_cached[:, i] for i in range(v_cached.shape[1])]
        window_k_write = [] if k_write_cached is None else [k_write_cached[:, i] for i in range(k_write_cached.shape[1])]
        return memory_state, window_k, window_v, window_k_write

    def _pack_cache_state(
        self,
        window_k: list[torch.Tensor],
        window_v: list[torch.Tensor],
        recent_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if recent_tokens is not None:
            recent_tokens = max(int(recent_tokens), 0)
            if recent_tokens == 0:
                raise ValueError("recent_tokens must be > 0 when packing cache state")
            window_k = window_k[-recent_tokens:]
            window_v = window_v[-recent_tokens:]
        k_window = torch.stack(window_k, dim=2)
        v_window = torch.stack(window_v, dim=2)
        k_window = k_window.transpose(1, 2).contiguous()
        v_window = v_window.transpose(1, 2).contiguous()
        k_flat = k_window.reshape(k_window.shape[0], k_window.shape[1], -1)
        v_flat = v_window.reshape(v_window.shape[0], v_window.shape[1], -1)
        return k_flat, v_flat

    def _run_fused_memory_scan(
        self,
        gates: torch.Tensor,
        updates: torch.Tensor,
        memory_state: torch.Tensor,
        *,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        updates = updates.float()
        if gates.shape[-1] == 1 and updates.shape[3] > 1:
            state_seq, final_state = self._run_shared_gate_memory_scan(
                gates,
                updates,
                memory_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )
        else:
            state_seq, final_state = self._run_per_slot_memory_scan(
                gates,
                updates,
                memory_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )
        return state_seq, final_state

    def _run_shared_gate_memory_scan(
        self,
        gates: torch.Tensor,
        updates: torch.Tensor,
        memory_state: torch.Tensor,
        *,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, num_heads, _ = gates.shape
        flat_gates = gates.squeeze(-1)
        flat_updates = updates.reshape(batch_size, seq_len, num_heads, self.num_mem_slots * self.head_dim)
        flat_state = memory_state.reshape(memory_state.shape[0], num_heads, self.num_mem_slots * self.head_dim).float()
        if gates.is_cuda:
            state_seq, final_state = fused_recurrent_gm_swa(
                x=flat_updates,
                g=flat_gates.float(),
                initial_state=flat_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                truncate_backward=True,
            )
        else:
            return self._run_memory_scan_torch(
                gates,
                updates,
                memory_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )

        state_seq = state_seq.view(batch_size, seq_len, num_heads, self.num_mem_slots, self.head_dim)
        if final_state is not None:
            final_state = final_state.view(memory_state.shape[0], num_heads, self.num_mem_slots, self.head_dim)
        return state_seq, final_state

    def _run_per_slot_memory_scan(
        self,
        gates: torch.Tensor,
        updates: torch.Tensor,
        memory_state: torch.Tensor,
        *,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, num_heads, num_slots = gates.shape
        flat_heads = num_heads * num_slots
        flat_gates = gates.reshape(batch_size, seq_len, flat_heads)
        flat_updates = updates.reshape(batch_size, seq_len, flat_heads, self.head_dim)
        if gates.is_cuda:
            flat_state = memory_state.reshape(memory_state.shape[0], flat_heads, self.head_dim).to(torch.float32)
            state_seq, final_state = fused_recurrent_gm_swa(
                x=flat_updates,
                g=flat_gates.float(),
                initial_state=flat_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                truncate_backward=True,
            )
        else:
            return self._run_memory_scan_torch(
                gates,
                updates,
                memory_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )

        state_seq = state_seq.view(batch_size, seq_len, num_heads, num_slots, self.head_dim)
        if final_state is not None:
            final_state = final_state.view(memory_state.shape[0], num_heads, num_slots, self.head_dim)
        return state_seq, final_state

    def _run_memory_scan_torch(
        self,
        gates: torch.Tensor,
        updates: torch.Tensor,
        memory_state: torch.Tensor,
        *,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, num_heads, num_slots = gates.shape
        flat_heads = num_heads * num_slots
        flat_gates = gates.reshape(batch_size, seq_len, flat_heads).float()
        flat_updates = updates.float().reshape(batch_size, seq_len, flat_heads, self.head_dim)
        flat_state = memory_state.reshape(memory_state.shape[0], flat_heads, self.head_dim).float()

        def affine_prefix_scan(
            seq_gates: torch.Tensor,
            seq_updates: torch.Tensor,
            init_state: torch.Tensor,
            *,
            seq_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            prefix_gates = seq_gates.clone()
            prefix_biases = ((1.0 - seq_gates).unsqueeze(-1) * seq_updates).clone()

            offset = 1
            while offset < seq_len:
                prev_gates = prefix_gates.clone()
                prev_biases = prefix_biases.clone()

                if seq_ids is None:
                    prefix_gates = torch.cat(
                        [
                            prev_gates[:, :offset],
                            prev_gates[:, offset:] * prev_gates[:, :-offset],
                        ],
                        dim=1,
                    )
                    prefix_biases = torch.cat(
                        [
                            prev_biases[:, :offset],
                            prev_gates[:, offset:].unsqueeze(-1) * prev_biases[:, :-offset] + prev_biases[:, offset:],
                        ],
                        dim=1,
                    )
                else:
                    valid = seq_ids[offset:] == seq_ids[:-offset]
                    if valid.any():
                        updated_gates = torch.where(
                            valid.view(1, -1, 1),
                            prev_gates[:, offset:] * prev_gates[:, :-offset],
                            prev_gates[:, offset:],
                        )
                        updated_biases = torch.where(
                            valid.view(1, -1, 1, 1),
                            prev_gates[:, offset:].unsqueeze(-1) * prev_biases[:, :-offset] + prev_biases[:, offset:],
                            prev_biases[:, offset:],
                        )
                        prefix_gates = torch.cat([prev_gates[:, :offset], updated_gates], dim=1)
                        prefix_biases = torch.cat([prev_biases[:, :offset], updated_biases], dim=1)
                    else:
                        prefix_gates = prev_gates
                        prefix_biases = prev_biases

                offset <<= 1

            if seq_ids is None:
                init_tokens = init_state.unsqueeze(1)
            else:
                init_tokens = init_state[seq_ids].unsqueeze(0)

            return prefix_gates.unsqueeze(-1) * init_tokens + prefix_biases

        if cu_seqlens is None:
            state_seq = affine_prefix_scan(flat_gates, flat_updates, flat_state)
            final_state = state_seq[:, -1]
            if not output_final_state:
                final_state = None
        else:
            seq_ids = prepare_sequence_ids(cu_seqlens)
            state_seq = affine_prefix_scan(flat_gates, flat_updates, flat_state, seq_ids=seq_ids)
            final_state = None
            if output_final_state:
                final_state = state_seq[:, cu_seqlens[1:] - 1].squeeze(0)

        state_seq = state_seq.view(batch_size, seq_len, num_heads, num_slots, self.head_dim)
        if final_state is not None:
            final_state = final_state.view(memory_state.shape[0], num_heads, num_slots, self.head_dim)
        return state_seq, final_state

    def _build_memory_inputs(
        self,
        hidden_states: torch.Tensor,
        k_kv: torch.Tensor,
        v_kv: torch.Tensor,
        *,
        valid_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _, _ = v_kv.shape
        device = v_kv.device
        dtype = v_kv.dtype
        gates = torch.ones(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.gate_num_slots,
            device=device,
            dtype=dtype,
        )
        updates = torch.zeros(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.num_mem_slots,
            self.head_dim,
            device=device,
            dtype=dtype,
        )

        if seq_len <= self.window_size:
            return gates, updates

        positions = torch.arange(seq_len, device=device)
        update_mask = torch.ones((batch_size, seq_len - self.window_size), device=device, dtype=torch.bool)
        if valid_tokens is not None:
            update_mask &= valid_tokens[:, self.window_size:]
        if self.mem_update_stride > 1:
            stride_mask = ((positions[self.window_size:] - self.window_size) % self.mem_update_stride) == 0
            update_mask &= stride_mask.unsqueeze(0)

        if not update_mask.any():
            return gates, updates

        gate_values = self._compute_gate(hidden_states[:, self.window_size:]).to(dtype=dtype)
        update_values = self._project_memory_update(
            k_kv[:, : seq_len - self.window_size],
            v_kv[:, : seq_len - self.window_size],
        ).to(dtype=dtype)
        selected_gates = gate_values[update_mask]
        self._accumulate_selection_metric("gate_mean", selected_gates)
        self._accumulate_selection_metric("gate_low_frac", (selected_gates < 0.1).float())
        self._accumulate_selection_metric("gate_high_frac", (selected_gates > 0.9).float())
        gates[:, self.window_size:] = torch.where(
            update_mask.unsqueeze(-1).unsqueeze(-1),
            gate_values,
            gates[:, self.window_size:],
        )
        updates[:, self.window_size:] = torch.where(
            update_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            update_values,
            updates[:, self.window_size:],
        )
        return gates, updates

    def _build_varlen_memory_inputs(
        self,
        hidden_states: torch.Tensor,
        k_kv: torch.Tensor,
        v_kv: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_tokens = hidden_states.shape[1]
        device = hidden_states.device
        dtype = hidden_states.dtype
        gates = torch.ones(
            1,
            total_tokens,
            self.num_kv_heads,
            self.gate_num_slots,
            device=device,
            dtype=dtype,
        )
        updates = torch.zeros(
            1,
            total_tokens,
            self.num_kv_heads,
            self.num_mem_slots,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        if total_tokens == 0:
            return gates, updates, hidden_states.new_zeros(0, dtype=torch.long)

        seq_ids = prepare_sequence_ids(cu_seqlens)
        pos_ids = prepare_position_ids(cu_seqlens)
        update_mask = pos_ids >= self.window_size
        if self.mem_update_stride > 1:
            update_mask &= ((pos_ids - self.window_size) % self.mem_update_stride) == 0

        if update_mask.any():
            gate_values = self._compute_gate(hidden_states.squeeze(0)[update_mask]).to(dtype=dtype)
            source_idx = cu_seqlens[seq_ids[update_mask]] + (pos_ids[update_mask] - self.window_size)
            update_values = self._project_memory_update(
                k_kv.squeeze(0)[source_idx],
                v_kv.squeeze(0)[source_idx],
            ).to(dtype=dtype)
            self._accumulate_selection_metric("gate_mean", gate_values)
            self._accumulate_selection_metric("gate_low_frac", (gate_values < 0.1).float())
            self._accumulate_selection_metric("gate_high_frac", (gate_values > 0.9).float())
            gates[0, update_mask] = gate_values
            updates[0, update_mask] = update_values

        return gates, updates, pos_ids

    def _memory_available_mask(
        self,
        positions: torch.Tensor,
        *,
        batch_size: int,
        has_prior_memory: bool,
        valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(batch_size, -1)

        if has_prior_memory:
            memory_available = torch.ones_like(positions, dtype=torch.bool)
        else:
            memory_available = positions >= self.window_size

        if valid_tokens is not None:
            memory_available &= valid_tokens
        if self.mem_token_threshold is not None:
            memory_available &= positions.add(1) >= self.mem_token_threshold
        return memory_available

    def _combine_memory_with_local(
        self,
        q: torch.Tensor,
        o_local: torch.Tensor,
        lse_local: torch.Tensor,
        memory_states: torch.Tensor,
        memory_available: torch.Tensor,
        *,
        valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = q.dtype
        memory_states = self._normalize_memory(memory_states.to(torch.float32)).to(dtype)
        self._debug_nonfinite("memory_states", memory_states)
        self._debug_large("memory_states", memory_states)
        mem_k, mem_v = self._memory_kv(memory_states)

        lse_local = lse_local.transpose(1, 2).to(torch.float32)
        memory_available = memory_available.unsqueeze(-1).expand(-1, -1, self.num_heads)
        valid_heads = None
        if valid_tokens is not None:
            valid_heads = valid_tokens.unsqueeze(-1).expand(-1, -1, self.num_heads)

        if memory_states.shape[2] == self.num_kv_heads and self.num_kv_groups > 1:
            batch_size, seq_len = q.shape[:2]
            q_grouped = q.view(batch_size, seq_len, self.num_kv_heads, self.num_kv_groups, self.head_dim)
            stats_mask = memory_available if valid_heads is None else (memory_available & valid_heads)
            stats_mask_grouped = stats_mask.view(batch_size, seq_len, self.num_kv_heads, self.num_kv_groups)
            if self.num_mem_slots == 1:
                mem_k = mem_k.squeeze(-2)
                mem_v = mem_v.squeeze(-2)
                mem_score = torch.einsum("bthgd,bthd->bthg", q_grouped, mem_k).reshape(batch_size, seq_len, self.num_heads)
                mem_score = mem_score.to(torch.float32) * self.scaling
                self._debug_large("mem_score", mem_score)
                mem_score = mem_score.masked_fill(~memory_available, float("-inf"))
                total_lse = torch.logaddexp(lse_local, mem_score)
                self._debug_large("mem_total_lse", total_lse)
                if valid_heads is not None:
                    total_lse = torch.where(valid_heads, total_lse, torch.zeros_like(total_lse))
                    local_weight = torch.where(valid_heads, torch.exp(lse_local - total_lse), torch.zeros_like(total_lse))
                    mem_weight = torch.where(valid_heads, torch.exp(mem_score - total_lse), torch.zeros_like(total_lse))
                else:
                    local_weight = torch.exp(lse_local - total_lse)
                    mem_weight = torch.exp(mem_score - total_lse)
                self._debug_large("mem_weight", mem_weight)
                self._accumulate_selection_metric("mem_weight_mean", mem_weight, stats_mask)
                mem_out = mem_v.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1).reshape(
                    batch_size,
                    seq_len,
                    self.num_heads,
                    self.head_dim,
                )
                out = local_weight.to(dtype).unsqueeze(-1) * o_local + mem_weight.to(dtype).unsqueeze(-1) * mem_out
            else:
                mem_logits = torch.einsum("bthgd,bthsd->bthgs", q_grouped, mem_k)
                mem_logits = mem_logits.to(torch.float32) * self.scaling
                self._debug_large("mem_logits", mem_logits)
                mem_lse = torch.logsumexp(mem_logits, dim=-1).reshape(batch_size, seq_len, self.num_heads)
                mem_lse = mem_lse - self.multi_slot_log_prior
                self._debug_large("mem_lse", mem_lse)
                mem_lse = mem_lse.masked_fill(~memory_available, float("-inf"))
                mem_probs = torch.softmax(mem_logits, dim=-1).to(dtype)
                slot_entropy = -(mem_probs.float() * mem_probs.float().clamp_min(self.mem_norm_eps).log()).sum(dim=-1)
                slot_entropy = slot_entropy / math.log(self.num_mem_slots)
                self._accumulate_selection_metric("slot_entropy", slot_entropy, stats_mask_grouped)
                mem_out = torch.einsum("bthgs,bthsd->bthgd", mem_probs, mem_v).reshape(
                    batch_size,
                    seq_len,
                    self.num_heads,
                    self.head_dim,
                )
                total_lse = torch.logaddexp(lse_local, mem_lse)
                self._debug_large("mem_total_lse", total_lse)
                if valid_heads is not None:
                    total_lse = torch.where(valid_heads, total_lse, torch.zeros_like(total_lse))
                    local_weight = torch.where(valid_heads, torch.exp(lse_local - total_lse), torch.zeros_like(total_lse))
                    mem_weight = torch.where(valid_heads, torch.exp(mem_lse - total_lse), torch.zeros_like(total_lse))
                else:
                    local_weight = torch.exp(lse_local - total_lse)
                    mem_weight = torch.exp(mem_lse - total_lse)
                self._debug_large("mem_weight", mem_weight)
                self._accumulate_selection_metric("mem_weight_mean", mem_weight, stats_mask)
                out = local_weight.to(dtype).unsqueeze(-1) * o_local + mem_weight.to(dtype).unsqueeze(-1) * mem_out
            if valid_tokens is not None:
                out = out.masked_fill(~valid_tokens.unsqueeze(-1).unsqueeze(-1), 0)
            self._debug_nonfinite("memory_out", out)
            self._debug_nonfinite_grad("memory_out", out)
            return out

        stats_mask = memory_available if valid_heads is None else (memory_available & valid_heads)
        if self.num_mem_slots == 1:
            mem_k = mem_k.squeeze(-2)
            mem_v = mem_v.squeeze(-2)
            mem_score = torch.einsum("bthd,bthd->bth", q, mem_k).to(torch.float32) * self.scaling
            self._debug_large("mem_score", mem_score)
            mem_score = mem_score.masked_fill(~memory_available, float("-inf"))
            total_lse = torch.logaddexp(lse_local, mem_score)
            self._debug_large("mem_total_lse", total_lse)
            if valid_heads is not None:
                total_lse = torch.where(valid_heads, total_lse, torch.zeros_like(total_lse))
                local_weight = torch.where(valid_heads, torch.exp(lse_local - total_lse), torch.zeros_like(total_lse))
                mem_weight = torch.where(valid_heads, torch.exp(mem_score - total_lse), torch.zeros_like(total_lse))
            else:
                local_weight = torch.exp(lse_local - total_lse)
                mem_weight = torch.exp(mem_score - total_lse)
            self._debug_large("mem_weight", mem_weight)
            self._accumulate_selection_metric("mem_weight_mean", mem_weight, stats_mask)
            out = local_weight.to(dtype).unsqueeze(-1) * o_local + mem_weight.to(dtype).unsqueeze(-1) * mem_v
        else:
            mem_logits = torch.einsum("bthd,bthsd->bths", q, mem_k).to(torch.float32) * self.scaling
            self._debug_large("mem_logits", mem_logits)
            mem_lse = torch.logsumexp(mem_logits, dim=-1) - self.multi_slot_log_prior
            self._debug_large("mem_lse", mem_lse)
            mem_lse = mem_lse.masked_fill(~memory_available, float("-inf"))
            mem_probs = torch.softmax(mem_logits, dim=-1).to(dtype)
            slot_entropy = -(mem_probs.float() * mem_probs.float().clamp_min(self.mem_norm_eps).log()).sum(dim=-1)
            slot_entropy = slot_entropy / math.log(self.num_mem_slots)
            self._accumulate_selection_metric("slot_entropy", slot_entropy, stats_mask)
            mem_out = torch.einsum("bths,bthsd->bthd", mem_probs, mem_v)
            total_lse = torch.logaddexp(lse_local, mem_lse)
            self._debug_large("mem_total_lse", total_lse)
            if valid_heads is not None:
                total_lse = torch.where(valid_heads, total_lse, torch.zeros_like(total_lse))
                local_weight = torch.where(valid_heads, torch.exp(lse_local - total_lse), torch.zeros_like(total_lse))
                mem_weight = torch.where(valid_heads, torch.exp(mem_lse - total_lse), torch.zeros_like(total_lse))
            else:
                local_weight = torch.exp(lse_local - total_lse)
                mem_weight = torch.exp(mem_lse - total_lse)
            self._debug_large("mem_weight", mem_weight)
            self._accumulate_selection_metric("mem_weight_mean", mem_weight, stats_mask)
            out = local_weight.to(dtype).unsqueeze(-1) * o_local + mem_weight.to(dtype).unsqueeze(-1) * mem_out

        if valid_tokens is not None:
            out = out.masked_fill(~valid_tokens.unsqueeze(-1).unsqueeze(-1), 0)
        self._debug_nonfinite("memory_out", out)
        self._debug_large("memory_out", out)
        self._debug_nonfinite_grad("memory_out", out)
        return out

    def _local_attention_step(
        self,
        q_t: torch.Tensor,
        k_window: torch.Tensor,
        v_window: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_scores = torch.einsum("bhd,bhsd->bhs", q_t.float(), k_window.float()) * self.scaling
        local_lse = torch.logsumexp(local_scores, dim=-1)
        local_probs = torch.softmax(local_scores, dim=-1).to(dtype=v_window.dtype)
        o_local = torch.einsum("bhs,bhsd->bhd", local_probs, v_window)
        return o_local, local_lse

    def _combine_single_memory_step(
        self,
        q_t: torch.Tensor,
        o_local: torch.Tensor,
        lse_local: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> torch.Tensor:
        dtype = q_t.dtype
        mem_state = self._normalize_memory(memory_state.to(torch.float32)).to(dtype)
        mem_k, mem_v = self._memory_kv_for_queries(mem_state)
        mem_k = mem_k.squeeze(-2)
        mem_v = mem_v.squeeze(-2)
        mem_score = torch.einsum("bhd,bhd->bh", q_t, mem_k).to(torch.float32) * self.scaling
        total_lse = torch.logaddexp(lse_local, mem_score)
        local_weight = torch.exp(lse_local - total_lse)
        mem_weight = torch.exp(mem_score - total_lse)
        self._accumulate_selection_metric("mem_weight_mean", mem_weight)
        return local_weight.to(dtype).unsqueeze(-1) * o_local + mem_weight.to(dtype).unsqueeze(-1) * mem_v

    def _combine_memory_step(
        self,
        q_t: torch.Tensor,
        o_local: torch.Tensor,
        lse_local: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_mem_slots == 1:
            return self._combine_single_memory_step(q_t, o_local, lse_local, memory_state)
        batch_size = q_t.shape[0]
        memory_available = torch.ones((batch_size, 1), device=q_t.device, dtype=torch.bool)
        return self._combine_memory_with_local(
            q_t.unsqueeze(1),
            o_local.unsqueeze(1),
            lse_local.unsqueeze(-1),
            memory_state.unsqueeze(1),
            memory_available,
        ).squeeze(1)

    def _memory_and_combine_padded(
        self,
        q: torch.Tensor,
        k_kv: torch.Tensor,
        v_kv: torch.Tensor,
        hidden_states: torch.Tensor,
        memory_state: torch.Tensor,
        o_local: torch.Tensor,
        lse_local: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        has_prior_memory: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _, _ = q.shape
        if seq_len == 0:
            return o_local, memory_state

        valid_tokens = torch.arange(seq_len, device=q.device).unsqueeze(0) < seq_lens.unsqueeze(1)
        gates, updates = self._build_memory_inputs(hidden_states, k_kv, v_kv, valid_tokens=valid_tokens)
        self._debug_nonfinite_grad("updates", updates)
        memory_states, final_state = self._run_fused_memory_scan(
            gates,
            updates,
            memory_state,
            output_final_state=True,
        )
        self._debug_nonfinite_grad("memory_states_scan", memory_states)
        memory_available = self._memory_available_mask(
            torch.arange(seq_len, device=q.device),
            batch_size=batch_size,
            has_prior_memory=has_prior_memory,
            valid_tokens=valid_tokens,
        )
        out = self._combine_memory_with_local(
            q,
            o_local,
            lse_local,
            memory_states,
            memory_available,
            valid_tokens=valid_tokens,
        )
        return out, final_state.to(dtype=q.dtype)

    def _memory_and_combine_dense(
        self,
        q: torch.Tensor,
        k_kv: torch.Tensor,
        v_kv: torch.Tensor,
        hidden_states: torch.Tensor,
        memory_state: torch.Tensor,
        o_local: torch.Tensor,
        lse_local: torch.Tensor,
        *,
        has_prior_memory: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _, _ = q.shape
        if seq_len == 0:
            return o_local, memory_state

        gates, updates = self._build_memory_inputs(hidden_states, k_kv, v_kv)
        self._debug_nonfinite_grad("updates", updates)
        memory_states, final_state = self._run_fused_memory_scan(
            gates,
            updates,
            memory_state,
            output_final_state=True,
        )
        self._debug_nonfinite_grad("memory_states_scan", memory_states)
        memory_available = self._memory_available_mask(
            torch.arange(seq_len, device=q.device),
            batch_size=batch_size,
            has_prior_memory=has_prior_memory,
        )
        out = self._combine_memory_with_local(q, o_local, lse_local, memory_states, memory_available)
        return out, final_state.to(dtype=q.dtype)

    def _memory_and_combine_varlen(
        self,
        q: torch.Tensor,
        k_kv: torch.Tensor,
        v_kv: torch.Tensor,
        hidden_states: torch.Tensor,
        o_local: torch.Tensor,
        lse_local: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        if cu_seqlens.ndim > 1:
            cu_seqlens = cu_seqlens.squeeze(0)

        num_seqs = cu_seqlens.numel() - 1
        memory_state = torch.zeros(
            (num_seqs, self.num_kv_heads, self.num_mem_slots, self.head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        gates, updates, pos_ids = self._build_varlen_memory_inputs(hidden_states, k_kv, v_kv, cu_seqlens)
        self._debug_nonfinite("gates", gates)
        self._debug_nonfinite("updates", updates)
        self._debug_large("gates", gates)
        self._debug_large("updates", updates)
        self._debug_nonfinite_grad("updates", updates)
        memory_states, _ = self._run_fused_memory_scan(
            gates,
            updates,
            memory_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        self._debug_nonfinite("memory_states_scan", memory_states)
        self._debug_large("memory_states_scan", memory_states)
        self._debug_nonfinite_grad("memory_states_scan", memory_states)
        memory_available = self._memory_available_mask(
            pos_ids,
            batch_size=1,
            has_prior_memory=False,
        )
        return self._combine_memory_with_local(q, o_local, lse_local, memory_states, memory_available)

    def _forward_fast(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if flash_attn_func is None:
            raise RuntimeError("flash-attn is required for fast GM-SWA training.")

        batch_size, seq_len, _ = hidden_states.shape
        original_seq_len = seq_len
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_kv_write = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v_kv = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        self._debug_nonfinite("q_proj", q)
        self._debug_nonfinite("k_proj", k_kv_write)
        self._debug_nonfinite("v_proj", v_kv)
        self._debug_large("q_proj", q)
        self._debug_large("k_proj", k_kv_write)
        self._debug_large("v_proj", v_kv)

        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            if torch.all(attention_mask):
                attention_mask = None

        k_cache = None
        v_cache = None
        k_write_cache = None
        scatter_info = None
        masked_seq_lens = None

        if attention_mask is not None:
            if flash_attn_varlen_func is None:
                raise RuntimeError("flash-attn varlen kernels are required for padding-mask GM-SWA.")
            q_unpad, (k_unpad, v_unpad), indices_q, cu_seqlens_fa, max_seq_lens = unpad_input(
                q, (k_kv_write, v_kv), attention_mask, seq_len
            )
            cu_q, cu_k = cu_seqlens_fa
            max_q, max_k = max_seq_lens
            q_rope, k_rope = self._apply_rope(
                q_unpad.unsqueeze(0),
                k_unpad.unsqueeze(0),
                cu_seqlens=cu_q,
                max_seqlen=q_unpad.shape[0],
            )
            q_unpad = q_rope.squeeze(0)
            k_unpad = k_rope.squeeze(0)
            k_local = repeat_kv(k_unpad.unsqueeze(0), self.num_kv_groups).squeeze(0)
            v_local = repeat_kv(v_unpad.unsqueeze(0), self.num_kv_groups).squeeze(0)
            local_out_unpad, local_lse_unpad = flash_attn_varlen_func(
                q_unpad,
                k_local,
                v_local,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=max_q,
                max_seqlen_k=max_k,
                causal=True,
                window_size=(self.window_size - 1, 0),
                return_attn_probs=True,
            )[:2]
            masked_seq_lens = prepare_lens_from_mask(attention_mask)
            compact_len = int(masked_seq_lens.max().item())
            valid_rows = torch.nonzero(attention_mask, as_tuple=False)
            seq_ids = valid_rows[:, 0]
            pos_ids = (attention_mask.cumsum(-1) - 1)[attention_mask]

            def pack_valid(x: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
                packed = x.new_full((batch_size, compact_len, *x.shape[1:]), fill_value)
                packed[seq_ids, pos_ids] = x
                return packed

            q = pack_valid(q_unpad)
            k_kv_write = pack_valid(k_unpad)
            v_kv = pack_valid(v_unpad)
            hidden_states = pack_valid(hidden_states[attention_mask])
            local_out = pack_valid(local_out_unpad)
            local_lse = local_lse_unpad.new_full((batch_size, compact_len, self.num_heads), float("-inf"))
            local_lse[seq_ids, pos_ids] = local_lse_unpad.permute(1, 0).contiguous().to(local_lse.dtype)
            local_lse = local_lse.permute(0, 2, 1).contiguous()
            scatter_info = (attention_mask, seq_ids, pos_ids)
            seq_len = compact_len
        elif cu_seqlens is not None:
            if flash_attn_varlen_func is None:
                raise RuntimeError("flash-attn varlen kernels are required for packed GM-SWA.")
            max_seq_len = int(prepare_lens(cu_seqlens).max().item())
            q, k_kv = self._apply_rope(q, k_kv_write, cu_seqlens=cu_seqlens, max_seqlen=q.shape[1])
            k_local = repeat_kv(k_kv, self.num_kv_groups)
            v_local = repeat_kv(v_kv, self.num_kv_groups)
            local_out, local_lse = flash_attn_varlen_func(
                q.squeeze(0),
                k_local.squeeze(0),
                v_local.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seq_len,
                max_seqlen_k=max_seq_len,
                causal=True,
                window_size=(self.window_size - 1, 0),
                return_attn_probs=True,
            )[:2]
            local_out = local_out.unsqueeze(0)
            local_lse = local_lse.unsqueeze(0)
            self._debug_nonfinite("local_out", local_out)
            self._debug_large("local_out", local_out)
            self._debug_nonfinite_grad("local_out", local_out)
        else:
            q, k_kv = self._apply_rope(q, k_kv_write)
            k_local = repeat_kv(k_kv, self.num_kv_groups)
            v_local = repeat_kv(v_kv, self.num_kv_groups)
            local_out, local_lse = flash_attn_func(
                q,
                k_local,
                v_local,
                causal=True,
                window_size=(self.window_size - 1, 0),
                return_attn_probs=True,
            )[:2]
            k_cache = k_local
            v_cache = v_local
            k_write_cache = k_kv_write
            self._debug_nonfinite("local_out", local_out)
            self._debug_large("local_out", local_out)
            self._debug_nonfinite_grad("local_out", local_out)

        if not self.memory_enabled:
            out = local_out
            memory_state = None
        else:
            memory_state = hidden_states.new_zeros(batch_size, self.num_kv_heads, self.num_mem_slots, self.head_dim)
            if masked_seq_lens is not None:
                out, memory_state = self._memory_and_combine_padded(
                    q,
                    k_kv_write,
                    v_kv,
                    hidden_states,
                    memory_state,
                    local_out,
                    local_lse,
                    masked_seq_lens,
                    has_prior_memory=False,
                )
            elif cu_seqlens is not None:
                out = self._memory_and_combine_varlen(
                    q,
                    k_kv_write,
                    v_kv,
                    hidden_states,
                    local_out,
                    local_lse,
                    cu_seqlens,
                )
            else:
                out, memory_state = self._memory_and_combine_dense(
                    q,
                    k_kv_write,
                    v_kv,
                    hidden_states,
                    memory_state,
                    local_out,
                    local_lse,
                    has_prior_memory=False,
                )

        out = self.o_proj(out.reshape(batch_size, seq_len, -1))
        self._debug_nonfinite("attn_out", out)
        self._debug_large("attn_out", out)
        self._debug_nonfinite_grad("attn_out", out)
        if scatter_info is not None:
            original_mask, seq_ids, pos_ids = scatter_info
            padded_out = out.new_zeros(batch_size, original_seq_len, out.shape[-1])
            padded_out[original_mask] = out[seq_ids, pos_ids]
            out = padded_out
        return out, memory_state, k_cache, v_cache, k_write_cache

    def _forward_projected(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        k_write: torch.Tensor,
        v: torch.Tensor,
        *,
        seqlen_offset: int = 0,
        memory_state: torch.Tensor | None = None,
        window_k: list[torch.Tensor] | None = None,
        window_v: list[torch.Tensor] | None = None,
        window_k_write: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape

        if window_k is None:
            window_k = []
        if window_v is None:
            window_v = []
        if window_k_write is None:
            window_k_write = []

        if self.memory_enabled and memory_state is None:
            memory_state = torch.zeros(
                (batch_size, self.num_kv_heads, self.num_mem_slots, self.head_dim),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        outputs = []
        for token_idx in range(seq_len):
            seen_tokens_before_append = seqlen_offset + token_idx
            if len(window_k) >= self.window_size:
                window_k.pop(0)
                evicted_v = window_v.pop(0)
                evicted_k_write = window_k_write.pop(0)
                if self.memory_enabled and self._should_update_memory(seen_tokens_before_append):
                    memory_state = self._update_memory(
                        memory_state,
                        evicted_k_write,
                        self._collapse_query_groups(evicted_v),
                        hidden_states[:, token_idx],
                    )

            window_k.append(k[:, token_idx])
            window_v.append(v[:, token_idx])
            window_k_write.append(k_write[:, token_idx])

            k_window = torch.stack(window_k, dim=2)
            v_window = torch.stack(window_v, dim=2)
            q_t = q[:, token_idx].unsqueeze(2)

            use_memory = memory_state is not None and self._should_use_memory(seen_tokens_before_append)

            if use_memory:
                o_local, lse_local = self._local_attention_step(q[:, token_idx], k_window, v_window)
                attn_out = self._combine_memory_step(q[:, token_idx], o_local, lse_local, memory_state)
                attn_out = attn_out.reshape(batch_size, 1, -1)
            else:
                attn_out = F.scaled_dot_product_attention(q_t, k_window, v_window, is_causal=False)
                attn_out = attn_out.squeeze(2).reshape(batch_size, 1, -1)
            outputs.append(self.o_proj(attn_out))

        return torch.cat(outputs, dim=1), memory_state, window_k, window_v, window_k_write

    def _forward_full(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        seqlen_offset: int = 0,
        memory_state: torch.Tensor | None = None,
        window_k: list[torch.Tensor] | None = None,
        window_v: list[torch.Tensor] | None = None,
        window_k_write: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if attention_mask is not None and not torch.all(attention_mask.bool()):
            raise ValueError("GatedMemSWA full fallback does not support padding masks; use flash-attn fast path.")

        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_write = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v_kv = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q, k = self._apply_rope(q, k_write, seqlen_offset=seqlen_offset)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v_kv, self.num_kv_groups)

        return self._forward_projected(
            hidden_states,
            q,
            k,
            k_write,
            v,
            seqlen_offset=seqlen_offset,
            memory_state=memory_state,
            window_k=window_k,
            window_v=window_v,
            window_k_write=window_k_write,
        )

    def _forward_cached(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache,
        *,
        seqlen_offset: int,
    ) -> tuple[torch.Tensor, Cache]:
        batch_size = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(batch_size, 1, self.num_heads, self.head_dim)
        k_write = self.k_proj(hidden_states).view(batch_size, 1, self.num_kv_heads, self.head_dim)
        v_kv = self.v_proj(hidden_states).view(batch_size, 1, self.num_kv_heads, self.head_dim)

        q, k = self._apply_rope(q, k_write, seqlen_offset=seqlen_offset, max_seqlen=seqlen_offset + 1)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v_kv, self.num_kv_groups)

        memory_state, k_cached, v_cached, k_write_cached = self._load_cached_state(
            past_key_values,
            batch_size,
            hidden_states.device,
            hidden_states.dtype,
        )
        if k_cached is None:
            k_cached = k.new_empty((batch_size, 0, self.num_heads, self.head_dim))
            v_cached = v.new_empty((batch_size, 0, self.num_heads, self.head_dim))
            k_write_cached = k_write.new_empty((batch_size, 0, self.num_kv_heads, self.head_dim))

        if k_cached.shape[1] >= self.window_size:
            evicted_k_write = k_write_cached[:, 0]
            evicted_v = v_cached[:, 0]
            k_cached = k_cached[:, 1:]
            v_cached = v_cached[:, 1:]
            k_write_cached = k_write_cached[:, 1:]
            if self.memory_enabled and self._should_update_memory(seqlen_offset):
                memory_state = self._update_memory(
                    memory_state,
                    evicted_k_write,
                    self._collapse_query_groups(evicted_v),
                    hidden_states[:, 0],
                )

        k_cached = torch.cat([k_cached, k], dim=1)
        v_cached = torch.cat([v_cached, v], dim=1)
        k_write_cached = torch.cat([k_write_cached, k_write], dim=1)

        past_key_values.update(
            recurrent_state=memory_state,
            attn_state=(k.reshape(batch_size, 1, -1), v.reshape(batch_size, 1, -1), k_write.reshape(batch_size, 1, -1)),
            layer_idx=self.layer_idx,
            offset=1,
            cache_kwargs=dict(window_size=self.window_size),
        )

        k_window = k_cached.transpose(1, 2).contiguous()
        v_window = v_cached.transpose(1, 2).contiguous()
        q_t = q[:, 0].unsqueeze(2)

        use_memory = memory_state is not None and self._should_use_memory(seqlen_offset)

        if use_memory:
            o_local, lse_local = self._local_attention_step(q[:, 0], k_window, v_window)
            attn_out = self._combine_memory_step(q[:, 0], o_local, lse_local, memory_state)
            attn_out = attn_out.reshape(batch_size, 1, -1)
        else:
            attn_out = F.scaled_dot_product_attention(q_t, k_window, v_window, is_causal=False)
            attn_out = attn_out.squeeze(2).reshape(batch_size, 1, -1)
        return self.o_proj(attn_out), past_key_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Any | None]:
        if output_attentions:
            output_attentions = False

        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            if torch.all(attention_mask):
                attention_mask = None

        cache_len = 0
        if past_key_values is not None:
            cache_len = past_key_values.get_seq_length(self.layer_idx)

        cu_seqlens = kwargs.get("cu_seqlens")
        if hidden_states.shape[1] > 1 and cache_len == 0:
            out, memory_state, k, v, k_write = self._forward_fast(
                hidden_states,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
            )
            if (
                use_cache
                and past_key_values is not None
                and attention_mask is None
                and cu_seqlens is None
                and k is not None
                and v is not None
                and k_write is not None
            ):
                window_k = [k[:, i] for i in range(max(0, k.shape[1] - self.window_size), k.shape[1])]
                window_v = [v[:, i] for i in range(max(0, v.shape[1] - self.window_size), v.shape[1])]
                window_k_write = [
                    k_write[:, i] for i in range(max(0, k_write.shape[1] - self.window_size), k_write.shape[1])
                ]
                k_flat, v_flat = self._pack_cache_state(window_k, window_v)
                k_write_flat, _ = self._pack_cache_state(window_k_write, window_k_write)
                past_key_values.update(
                    recurrent_state=memory_state,
                    attn_state=(k_flat, v_flat, k_write_flat),
                    layer_idx=self.layer_idx,
                    offset=hidden_states.shape[1],
                    cache_kwargs=dict(window_size=self.window_size),
                )
            return out, None, past_key_values if use_cache else None

        if use_cache and past_key_values is not None and hidden_states.shape[1] == 1:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            out, past_key_values = self._forward_cached(hidden_states, past_key_values, seqlen_offset=seqlen_offset)
            return out, None, past_key_values

        memory_state, window_k, window_v, window_k_write = self._prepare_cache_state(
            past_key_values if use_cache else None,
            hidden_states.shape[0],
            hidden_states.device,
            hidden_states.dtype,
        )
        seqlen_offset = past_key_values.get_seq_length(self.layer_idx) if use_cache and past_key_values is not None else 0
        out, memory_state, window_k, window_v, window_k_write = self._forward_full(
            hidden_states,
            attention_mask=attention_mask,
            seqlen_offset=seqlen_offset,
            memory_state=memory_state,
            window_k=window_k,
            window_v=window_v,
            window_k_write=window_k_write,
        )

        if use_cache and past_key_values is not None:
            k_flat, v_flat = self._pack_cache_state(
                window_k,
                window_v,
                recent_tokens=min(hidden_states.shape[1], self.window_size),
            )
            k_write_flat, _ = self._pack_cache_state(
                window_k_write,
                window_k_write,
                recent_tokens=min(hidden_states.shape[1], self.window_size),
            )
            past_key_values.update(
                recurrent_state=memory_state,
                attn_state=(k_flat, v_flat, k_write_flat),
                layer_idx=self.layer_idx,
                offset=hidden_states.shape[1],
                cache_kwargs=dict(window_size=self.window_size),
            )

        return out, None, past_key_values if use_cache else None

    def inference_step(
        self,
        x_t: torch.Tensor,
        kv_cache: dict[str, Any] | None,
        memory_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor | None]:
        if self.window_size is None:
            raise ValueError("inference_step requires window_size")

        batch_size = x_t.shape[0]
        if kv_cache is None:
            kv_cache = {"k": None, "v": None, "idx": 0, "filled": 0, "seen": 0}
        if self.memory_enabled and memory_state is None:
            memory_state = torch.zeros(
                (batch_size, self.num_kv_heads, self.num_mem_slots, self.head_dim),
                device=x_t.device,
                dtype=x_t.dtype,
            )

        q = self.q_proj(x_t).view(batch_size, 1, self.num_heads, self.head_dim)
        k = self.k_proj(x_t).view(batch_size, 1, self.num_kv_heads, self.head_dim)
        v_kv = self.v_proj(x_t).view(batch_size, 1, self.num_kv_heads, self.head_dim)

        seqlen_offset = int(kv_cache.get("seen", 0))
        q, k = self._apply_rope(q, k, seqlen_offset=seqlen_offset, max_seqlen=seqlen_offset + 1)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v_kv, self.num_kv_groups)

        if kv_cache["k"] is None:
            kv_cache["k"] = torch.zeros(
                (batch_size, self.window_size, self.num_heads, self.head_dim),
                device=x_t.device,
                dtype=x_t.dtype,
            )
            kv_cache["v"] = torch.zeros_like(kv_cache["k"])

        if kv_cache["filled"] >= self.window_size and self.memory_enabled:
            evicted_k = kv_cache["k"][:, kv_cache["idx"]]
            evicted_v = kv_cache["v"][:, kv_cache["idx"]]
            if self._should_update_memory(seqlen_offset):
                memory_state = self._update_memory(
                    memory_state,
                    self._collapse_query_groups(evicted_k),
                    self._collapse_query_groups(evicted_v),
                    x_t[:, 0],
                )

        kv_cache["k"][:, kv_cache["idx"]] = k[:, 0]
        kv_cache["v"][:, kv_cache["idx"]] = v[:, 0]
        kv_cache["filled"] = min(self.window_size, kv_cache["filled"] + 1)
        kv_cache["idx"] = (kv_cache["idx"] + 1) % self.window_size
        kv_cache["seen"] = seqlen_offset + 1

        if kv_cache["filled"] < self.window_size:
            k_seq = kv_cache["k"][:, :kv_cache["filled"]].transpose(1, 2)
            v_seq = kv_cache["v"][:, :kv_cache["filled"]].transpose(1, 2)
        else:
            idx = kv_cache["idx"]
            k_seq = torch.cat([kv_cache["k"][:, idx:], kv_cache["k"][:, :idx]], dim=1).transpose(1, 2)
            v_seq = torch.cat([kv_cache["v"][:, idx:], kv_cache["v"][:, :idx]], dim=1).transpose(1, 2)

        q_t = q[:, 0].unsqueeze(2)
        use_memory = memory_state is not None and self._should_use_memory(seqlen_offset)

        if use_memory:
            o_local, lse_local = self._local_attention_step(q[:, 0], k_seq, v_seq)
            attn_out = self._combine_memory_step(q[:, 0], o_local, lse_local, memory_state)
            attn_out = attn_out.reshape(batch_size, 1, -1)
        else:
            attn_out = F.scaled_dot_product_attention(q_t, k_seq, v_seq, is_causal=False)
            attn_out = attn_out.squeeze(2).reshape(batch_size, 1, -1)
        return self.o_proj(attn_out), kv_cache, memory_state
