# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Falling back to SDPA for GM-SWA local attention.",
        category=ImportWarning,
    )
    flash_attn_func = None
    flash_attn_varlen_func = None

from fla.modules import RotaryEmbedding
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.utils.index import prepare_lens, prepare_position_ids

if TYPE_CHECKING:
    from fla.models.utils import Cache


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, seq_len, num_kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, seq_len, n_rep, num_kv_heads, head_dim)
    return hidden_states.reshape(batch, seq_len, num_kv_heads * n_rep, head_dim)


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class GatedMemSWA(nn.Module):
    """
    GM-SWA v2: sliding-window attention plus a gated-delta fast-weight memory.

    The memory state is a matrix-valued fast weight per query head with shape
    ``[B, H_q, d_h, d_h]``. It is updated from the pre-RoPE key/value pair that
    just fell out of the local window, then read with the current pre-RoPE query.
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
        disable_memory: bool = False,
        mem_gate_logit_bias: float = -2.0,
        mix_gate_logit_bias: float = 4.0,
        a_log_init_lo: float = 1.0,
        a_log_init_hi: float = 16.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 0.0001,
        layer_idx: int | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        super().__init__()
        ignored = {
            k: v
            for k, v in legacy_kwargs.items()
            if k
            in {
                "num_mem_slots",
                "num_memory_components",
                "use_memory_component",
                "memory_state_rank",
                "mem_scale",
                "mem_rank",
                "mem_proj_mode",
                "mem_gate_mode",
                "mem_update_source",
                "mem_update_stride",
                "mem_token_threshold",
                "gate_bias_init",
                "mem_norm",
                "mem_norm_eps",
            }
        }
        if ignored:
            warnings.warn(
                "GM-SWA v2 ignores legacy v1 memory-slot kwargs: "
                + ", ".join(sorted(ignored)),
                stacklevel=2,
            )

        if dim % num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}")
        if window_size <= 0:
            raise ValueError("window_size must be > 0 for GatedMemSWA")
        if a_log_init_lo <= 0 or a_log_init_hi <= 0 or a_log_init_hi < a_log_init_lo:
            raise ValueError("a_log_init_lo/a_log_init_hi must define a positive increasing range")
        if dt_min <= 0 or dt_max <= 0 or dt_max < dt_min:
            raise ValueError("dt_min/dt_max must define a positive increasing range")

        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.max_position_embeddings = max_position_embeddings
        self.disable_memory = disable_memory
        self.memory_enabled = not disable_memory
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=rope_theta)

        if self.memory_enabled:
            self.gate_proj = nn.Linear(dim, 3 * self.num_heads, bias=True)

            A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(a_log_init_lo, a_log_init_hi)
            self.A_log = nn.Parameter(torch.log(A))
            self.A_log._no_weight_decay = True

            dt = torch.exp(
                torch.rand(self.num_heads, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            self.dt_bias = nn.Parameter(_inverse_softplus(dt))
            self.dt_bias._no_weight_decay = True

            self.mem_gate_logit_bias = float(mem_gate_logit_bias)
            self.mix_gate_logit_bias = float(mix_gate_logit_bias)
            self.reset_memory_parameters()
        else:
            self.gate_proj = None
            self.A_log = None
            self.dt_bias = None
            self.mem_gate_logit_bias = float(mem_gate_logit_bias)
            self.mix_gate_logit_bias = float(mix_gate_logit_bias)

    @property
    def beta_proj(self) -> nn.Linear | None:
        return self.gate_proj

    def reset_memory_parameters(self) -> None:
        if not self.memory_enabled:
            return
        with torch.no_grad():
            nn.init.zeros_(self.gate_proj.weight)
            self.gate_proj.bias[: self.num_heads].fill_(self.mem_gate_logit_bias)
            self.gate_proj.bias[self.num_heads : 2 * self.num_heads].zero_()
            self.gate_proj.bias[2 * self.num_heads :].fill_(self.mix_gate_logit_bias)

    def _project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        return q, k, v

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        seqlen_offset: int | torch.Tensor = 0,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary(q, k, seqlen_offset=seqlen_offset, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    def _local_attention_dense(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if flash_attn_func is not None and q.dtype in {torch.float16, torch.bfloat16}:
            return flash_attn_func(q, k, v, causal=True, window_size=(self.window_size - 1, 0))

        seq_len = q.shape[1]
        device = q.device
        row = torch.arange(seq_len, device=device)[:, None]
        col = torch.arange(seq_len, device=device)[None, :]
        mask = (col <= row) & (col >= row - self.window_size + 1)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
            dropout_p=0.0,
        )
        return out.transpose(1, 2)

    def _local_attention_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        cu_seqlens = self._normalize_cu_seqlens(cu_seqlens)
        if flash_attn_varlen_func is not None and q.dtype in {torch.float16, torch.bfloat16}:
            max_seq_len = int(prepare_lens(cu_seqlens).max().item())
            out = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seq_len,
                max_seqlen_k=max_seq_len,
                causal=True,
                window_size=(self.window_size - 1, 0),
            )
            return out.unsqueeze(0)

        pieces = []
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False):
            pieces.append(self._local_attention_dense(q[:, start:end], k[:, start:end], v[:, start:end]))
        return torch.cat(pieces, dim=1) if pieces else q.new_empty(q.shape)

    def _local_attention_decode(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
        return out.transpose(1, 2)

    @staticmethod
    def _normalize_cu_seqlens(cu_seqlens: torch.Tensor | None) -> torch.Tensor | None:
        if cu_seqlens is None:
            return None
        if cu_seqlens.ndim > 1:
            cu_seqlens = cu_seqlens.squeeze(0)
        return cu_seqlens.to(dtype=torch.int32)

    def _gate_values(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gates = self.gate_proj(hidden_states)
        beta_logits, a_logits, mix_logits = gates.split(self.num_heads, dim=-1)
        beta = torch.sigmoid(beta_logits)
        g = -self.A_log.float().exp().view(1, 1, self.num_heads) * F.softplus(
            a_logits.float() + self.dt_bias.view(1, 1, self.num_heads)
        )
        alpha = torch.sigmoid(mix_logits)
        return beta, g, alpha

    def _shift_evicted_dense(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_e = torch.zeros_like(k)
        v_e = torch.zeros_like(v)
        valid = torch.zeros(k.shape[:2], device=k.device, dtype=torch.bool)
        if k.shape[1] > self.window_size:
            k_e[:, self.window_size :] = k[:, : -self.window_size]
            v_e[:, self.window_size :] = v[:, : -self.window_size]
            valid[:, self.window_size :] = True
        return k_e, v_e, valid

    def _shift_evicted_varlen(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cu_seqlens = self._normalize_cu_seqlens(cu_seqlens)
        total_len = k.shape[1]
        idx = torch.arange(total_len, device=k.device)
        pos = prepare_position_ids(cu_seqlens).to(device=k.device)
        valid_1d = pos >= self.window_size
        src = idx - self.window_size
        k_e = torch.zeros_like(k)
        v_e = torch.zeros_like(v)
        if valid_1d.any():
            k_e[:, valid_1d] = k[:, src[valid_1d]]
            v_e[:, valid_1d] = v[:, src[valid_1d]]
        return k_e, v_e, valid_1d.unsqueeze(0)

    def _memory_branch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.memory_enabled:
            return torch.zeros_like(q), None

        if k.shape[2] != q.shape[2]:
            if q.shape[2] % k.shape[2] != 0:
                raise ValueError(f"Cannot repeat memory K/V heads from {k.shape[2]} to {q.shape[2]}")
            n_rep = q.shape[2] // k.shape[2]
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)

        beta, g, _ = self._gate_values(hidden_states)
        if cu_seqlens is None:
            k_e, v_e, valid = self._shift_evicted_dense(k, v)
        else:
            k_e, v_e, valid = self._shift_evicted_varlen(k, v, cu_seqlens)
            cu_seqlens = self._normalize_cu_seqlens(cu_seqlens)

        beta = torch.where(valid.unsqueeze(-1), beta, torch.zeros_like(beta))
        g = torch.where(valid.unsqueeze(-1), g, torch.zeros_like(g))
        mode = "chunk" if self.training or q.shape[1] > 64 else "fused_recurrent"
        fn = chunk_gated_delta_rule if mode == "chunk" else fused_recurrent_gated_delta_rule
        out, final_state = fn(
            q=q,
            k=k_e,
            v=v_e,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            scale=1.0,
            use_qk_l2norm_in_kernel=True,
        )
        if final_state is not None:
            final_state = final_state.float()
        return out, final_state

    def _new_memory_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=torch.float32)

    def _inline_delta_rule_step(
        self,
        q_t: torch.Tensor,
        k_e: torch.Tensor,
        v_e: torch.Tensor,
        beta_t: torch.Tensor,
        g_t: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = state.float()
        q_hat = F.normalize(q_t.float(), p=2, dim=-1)
        k_hat = F.normalize(k_e.float(), p=2, dim=-1)
        state = state * torch.exp(g_t.float()).view(g_t.shape[0], g_t.shape[1], 1, 1)
        pred = torch.einsum("bhkv,bhk->bhv", state, k_hat)
        err = v_e.float() - pred
        state = state + beta_t.float().view(beta_t.shape[0], beta_t.shape[1], 1, 1) * torch.einsum(
            "bhk,bhv->bhkv",
            k_hat,
            err,
        )
        out = torch.einsum("bhkv,bhk->bhv", state, q_hat).to(dtype=q_t.dtype)
        return out, state

    def _forward_training_or_prefill(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: Cache | None = None,
    ) -> tuple[torch.Tensor, Cache | None]:
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            if not torch.all(attention_mask):
                raise RuntimeError("GM-SWA v2 currently expects packed varlen input instead of padded attention masks.")
            attention_mask = None

        q_pre, k_pre, v_pre = self._project_qkv(hidden_states)
        k_mem = repeat_kv(k_pre, self.num_kv_groups)
        v_mem = repeat_kv(v_pre, self.num_kv_groups)

        if cu_seqlens is not None:
            cu_seqlens = self._normalize_cu_seqlens(cu_seqlens)
            max_seq_len = max(
                int(prepare_lens(cu_seqlens).max().item()),
                q_pre.shape[1],
                self.max_position_embeddings or 0,
            )
            q_rope, k_rope = self._apply_rope(q_pre, k_pre, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)
            k_local = repeat_kv(k_rope, self.num_kv_groups)
            v_local = v_mem
            local_out = self._local_attention_varlen(q_rope, k_local, v_local, cu_seqlens)
        else:
            q_rope, k_rope = self._apply_rope(q_pre, k_pre)
            k_local = repeat_kv(k_rope, self.num_kv_groups)
            v_local = v_mem
            local_out = self._local_attention_dense(q_rope, k_local, v_local)

        if self.memory_enabled:
            mem_out, memory_state = self._memory_branch(
                q_pre,
                k_mem,
                v_mem,
                hidden_states,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            _, _, alpha = self._gate_values(hidden_states)
            out = alpha.unsqueeze(-1).to(local_out.dtype) * local_out + (1.0 - alpha.unsqueeze(-1).to(local_out.dtype)) * mem_out
        else:
            out = local_out
            memory_state = None

        out = self.o_proj(out.reshape(out.shape[0], out.shape[1], self.num_heads * self.head_dim))

        if use_cache and past_key_values is not None and attention_mask is None and cu_seqlens is None:
            k_cache = k_local[:, -self.window_size :].contiguous()
            v_cache = v_local[:, -self.window_size :].contiguous()
            k_write_cache = k_mem[:, -self.window_size :].contiguous()
            past_key_values.update(
                recurrent_state=memory_state,
                attn_state=(k_cache, v_cache, k_write_cache),
                layer_idx=self.layer_idx,
                offset=hidden_states.shape[1],
                cache_kwargs=dict(window_size=self.window_size),
            )

        return out, past_key_values

    def _forward_decode(self, hidden_states: torch.Tensor, past_key_values: Cache) -> tuple[torch.Tensor, Cache]:
        q_pre, k_pre, v_pre = self._project_qkv(hidden_states)
        seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
        q_rope, k_rope = self._apply_rope(q_pre, k_pre, seqlen_offset=seqlen_offset)
        k_current = repeat_kv(k_rope, self.num_kv_groups)
        v_current = repeat_kv(v_pre, self.num_kv_groups)
        k_write_current = repeat_kv(k_pre, self.num_kv_groups)

        try:
            state = past_key_values[self.layer_idx]
        except KeyError:
            state = None
        recurrent_state = None if state is None else state.get("recurrent_state")
        attn_state = None if state is None else state.get("attn_state")
        if recurrent_state is None and self.memory_enabled:
            recurrent_state = self._new_memory_state(hidden_states.shape[0], hidden_states.device)

        if attn_state is None:
            old_k = old_v = old_k_write = None
        else:
            old_k, old_v, old_k_write = attn_state

        if old_k is not None and old_k.shape[1] >= self.window_size:
            k_e = old_k_write[:, 0]
            v_e = old_v[:, 0]
            local_k = torch.cat([old_k[:, 1:], k_current], dim=1)
            local_v = torch.cat([old_v[:, 1:], v_current], dim=1)
        elif old_k is not None:
            k_e = torch.zeros_like(k_write_current[:, 0])
            v_e = torch.zeros_like(v_current[:, 0])
            local_k = torch.cat([old_k, k_current], dim=1)
            local_v = torch.cat([old_v, v_current], dim=1)
        else:
            k_e = torch.zeros_like(k_write_current[:, 0])
            v_e = torch.zeros_like(v_current[:, 0])
            local_k = k_current
            local_v = v_current

        local_out = self._local_attention_decode(q_rope, local_k, local_v)
        if self.memory_enabled:
            beta, g, alpha = self._gate_values(hidden_states)
            valid = old_k is not None and old_k.shape[1] >= self.window_size
            if not valid:
                beta = torch.zeros_like(beta)
                g = torch.zeros_like(g)
            mem_out, recurrent_state = self._inline_delta_rule_step(
                q_pre[:, 0],
                k_e,
                v_e,
                beta[:, 0],
                g[:, 0],
                recurrent_state,
            )
            mem_out = mem_out.unsqueeze(1)
            out = alpha.unsqueeze(-1).to(local_out.dtype) * local_out + (1.0 - alpha.unsqueeze(-1).to(local_out.dtype)) * mem_out
        else:
            out = local_out
            recurrent_state = None

        out = self.o_proj(out.reshape(out.shape[0], out.shape[1], self.num_heads * self.head_dim))
        past_key_values.update(
            recurrent_state=recurrent_state,
            attn_state=(k_current, v_current, k_write_current),
            layer_idx=self.layer_idx,
            offset=hidden_states.shape[1],
            cache_kwargs=dict(window_size=self.window_size),
        )
        return out, past_key_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if output_attentions:
            output_attentions = False
        cu_seqlens = self._normalize_cu_seqlens(kwargs.get("cu_seqlens"))

        if use_cache and past_key_values is not None and hidden_states.shape[1] == 1:
            out, past_key_values = self._forward_decode(hidden_states, past_key_values)
            return out, None, past_key_values

        out, past_key_values = self._forward_training_or_prefill(
            hidden_states,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        return out, None, past_key_values if use_cache else None
