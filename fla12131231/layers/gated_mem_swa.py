# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

try:
    from fla12131231.ops.utils.cumsum import chunk_global_cumsum_scalar, chunk_global_cumsum_vector
except ImportError:
    chunk_global_cumsum_scalar = None
    chunk_global_cumsum_vector = None

from fla12131231.modules import RotaryEmbedding


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
        mem_scale: float = 1.0,
        mem_rank: int | None = None,
        mem_proj_mode: str = "linear",
        mem_gate_mode: str = "linear",
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
        self.mem_rank = mem_rank
        self.mem_proj_mode = mem_proj_mode
        self.mem_gate_mode = mem_gate_mode
        self.mem_update_stride = mem_update_stride
        self.mem_token_threshold = mem_token_threshold
        self.disable_memory = disable_memory
        mem_scale = float(mem_scale)
        if mem_scale <= 0:
            raise ValueError("mem_scale must be > 0 when using learnable scale.")
        if mem_proj_mode not in {"linear", "scale"}:
            raise ValueError(f"Unsupported mem_proj_mode: {mem_proj_mode}")
        if mem_gate_mode not in {"linear", "param"}:
            raise ValueError(f"Unsupported mem_gate_mode: {mem_gate_mode}")
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

        if self.mem_gate_mode == "linear":
            self.gate_net = nn.Linear(dim, self.num_heads, bias=True)
            nn.init.constant_(self.gate_net.bias, gate_bias_init)
            self.gate_param = None
        else:
            self.gate_net = None
            self.gate_param = nn.Parameter(torch.full((self.num_heads,), float(gate_bias_init)))

        if self.mem_proj_mode == "linear":
            if self.mem_rank is None:
                self.mem_proj = nn.Linear(dim, dim, bias=False)
                self.mem_proj_in = None
                self.mem_proj_out = None
            else:
                self.mem_proj = None
                self.mem_proj_in = nn.Linear(dim, self.mem_rank, bias=False)
                self.mem_proj_out = nn.Linear(self.mem_rank, dim, bias=False)
            self.mem_proj_scale = None
        else:
            self.mem_proj = None
            self.mem_proj_in = None
            self.mem_proj_out = None
            self.mem_proj_scale = nn.Parameter(torch.ones(self.num_heads, self.head_dim))

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=rope_theta)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        seqlen_offset: int = 0,
        max_seqlen: int | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_seqlen is None:
            max_seqlen = seqlen_offset + q.shape[1]
        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        return self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

    def _normalize_memory(self, memory_state: torch.Tensor) -> torch.Tensor:
        if not self.mem_norm:
            return memory_state
        return F.normalize(memory_state, dim=-1, eps=self.mem_norm_eps)

    def _memory_context(self, memory_state: torch.Tensor, seen_tokens: int) -> torch.Tensor:
        memory_state = self._normalize_memory(memory_state)
        if self.mem_token_threshold is None:
            return memory_state
        if torch.is_tensor(seen_tokens):
            mask = seen_tokens < self.mem_token_threshold
            return torch.where(mask, torch.zeros_like(memory_state), memory_state)
        if seen_tokens < self.mem_token_threshold:
            return torch.zeros_like(memory_state)
        return memory_state

    def _memory_kv(self, memory_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return memory_state, memory_state * self.mem_scale

    @property
    def mem_scale(self) -> torch.Tensor:
        return torch.exp(self.log_mem_scale).clamp_min(self.mem_norm_eps)

    def _compute_gate(self, gate_input: torch.Tensor) -> torch.Tensor:
        if self.mem_gate_mode == "linear":
            return torch.sigmoid(self.gate_net(gate_input))
        gate = self.gate_param
        if gate_input.dim() == 3:
            gate = gate.view(1, 1, -1).expand(gate_input.shape[0], gate_input.shape[1], -1)
        else:
            gate = gate.view(1, -1).expand(gate_input.shape[0], -1)
        return torch.sigmoid(gate)

    def _project_memory_update(self, evicted_v: torch.Tensor) -> torch.Tensor:
        if self.mem_proj_mode == "scale":
            scale = self.mem_proj_scale.to(device=evicted_v.device, dtype=evicted_v.dtype)
            while scale.dim() < evicted_v.dim():
                scale = scale.unsqueeze(0)
            return evicted_v * scale

        evicted_flat = evicted_v.reshape(*evicted_v.shape[:-2], -1)
        if self.mem_proj is not None:
            mem_update = self.mem_proj(evicted_flat)
        else:
            mem_update = self.mem_proj_out(self.mem_proj_in(evicted_flat))
        return mem_update.view(*evicted_v.shape[:-2], self.num_heads, self.head_dim)

    def _should_update_memory(self, seen_tokens_before_append: int) -> bool:
        if seen_tokens_before_append < self.window_size:
            return False
        evicted_idx = seen_tokens_before_append - self.window_size
        return evicted_idx % self.mem_update_stride == 0

    def _update_memory(self, memory_state: torch.Tensor, evicted_v: torch.Tensor, gate_input: torch.Tensor) -> torch.Tensor:
        gate = self._compute_gate(gate_input).unsqueeze(-1)
        mem_update = self._project_memory_update(evicted_v)
        memory_state = gate * memory_state + (1.0 - gate) * mem_update
        return memory_state

    def _prepare_cache_state(
        self,
        past_key_values: Any | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        memory_state = torch.zeros((batch_size, self.num_heads, self.head_dim), device=device, dtype=dtype)
        window_k: list[torch.Tensor] = []
        window_v: list[torch.Tensor] = []

        if past_key_values is None or self.layer_idx is None:
            return memory_state, window_k, window_v
        if len(past_key_values) <= self.layer_idx:
            return memory_state, window_k, window_v

        state = past_key_values[self.layer_idx]
        if state is None:
            return memory_state, window_k, window_v

        cached_mem = state.get("recurrent_state")
        if cached_mem is not None:
            memory_state = cached_mem

        attn_state = state.get("attn_state")
        if attn_state is not None:
            k_cached, v_cached = attn_state
            k_cached = k_cached.view(batch_size, -1, self.num_heads, self.head_dim)
            v_cached = v_cached.view(batch_size, -1, self.num_heads, self.head_dim)
            window_k = [k_cached[:, i] for i in range(k_cached.shape[1])]
            window_v = [v_cached[:, i] for i in range(v_cached.shape[1])]

        return memory_state, window_k, window_v

    def _pack_cache_state(self, window_k: list[torch.Tensor], window_v: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        k_window = torch.stack(window_k, dim=2)
        v_window = torch.stack(window_v, dim=2)
        k_window = k_window.transpose(1, 2).contiguous()
        v_window = v_window.transpose(1, 2).contiguous()
        k_flat = k_window.reshape(k_window.shape[0], k_window.shape[1], -1)
        v_flat = v_window.reshape(v_window.shape[0], v_window.shape[1], -1)
        return k_flat, v_flat

    def _forward_projected(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        seqlen_offset: int = 0,
        memory_state: torch.Tensor | None = None,
        window_k: list[torch.Tensor] | None = None,
        window_v: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape

        if memory_state is None:
            memory_state = torch.zeros((batch_size, self.num_heads, self.head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
        if window_k is None:
            window_k = []
        if window_v is None:
            window_v = []

        outputs = []
        for t in range(seq_len):
            seen_tokens_before_append = seqlen_offset + t
            if len(window_k) >= self.window_size:
                evicted_v = window_v.pop(0)
                window_k.pop(0)
                if not self.disable_memory and self._should_update_memory(seen_tokens_before_append):
                    memory_state = self._update_memory(memory_state, evicted_v, hidden_states[:, t])

            window_k.append(k[:, t])
            window_v.append(v[:, t])

            k_window = torch.stack(window_k, dim=2)
            v_window = torch.stack(window_v, dim=2)
            q_t = q[:, t].unsqueeze(2)
            if self.disable_memory:
                k_cat = k_window
                v_cat = v_window
            else:
                mem_k, mem_v = self._memory_kv(self._memory_context(memory_state, seen_tokens_before_append + 1))
                k_cat = torch.cat([mem_k.unsqueeze(2), k_window], dim=2)
                v_cat = torch.cat([mem_v.unsqueeze(2), v_window], dim=2)

            attn_out = F.scaled_dot_product_attention(q_t, k_cat, v_cat, is_causal=False)
            attn_out = attn_out.squeeze(2).reshape(batch_size, 1, -1)
            outputs.append(self.o_proj(attn_out))

        return torch.cat(outputs, dim=1), memory_state, window_k, window_v

    def _forward_full(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        seqlen_offset: int = 0,
        memory_state: torch.Tensor | None = None,
        window_k: list[torch.Tensor] | None = None,
        window_v: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        if attention_mask is not None and not torch.all(attention_mask.bool()):
            raise ValueError("GatedMemSWA does not support padding masks yet.")

        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q, k = self._apply_rope(q, k, seqlen_offset=seqlen_offset)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        return self._forward_projected(
            hidden_states,
            q,
            k,
            v,
            seqlen_offset=seqlen_offset,
            memory_state=memory_state,
            window_k=window_k,
            window_v=window_v,
        )

    def _forward_fast(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        seqlen_offset: int = 0,
        memory_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if attention_mask is not None and not torch.all(attention_mask.bool()):
            raise ValueError("GatedMemSWA does not support padding masks yet.")

        if flash_attn_func is None:
            raise RuntimeError("flash-attn is required for fast GM-SWA training.")

        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q, k = self._apply_rope(q, k, seqlen_offset=seqlen_offset)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        if self.disable_memory:
            out = flash_attn_func(
                q,
                k,
                v,
                causal=True,
                window_size=(self.window_size - 1, 0),
            )
            out = out.reshape(batch_size, seq_len, -1)
            out = self.o_proj(out)
            memory_state = torch.zeros(
                (batch_size, self.num_heads, self.head_dim),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            return out, memory_state, k, v

        local_out, local_lse, _ = flash_attn_func(
            q,
            k,
            v,
            causal=True,
            window_size=(self.window_size - 1, 0),
            return_attn_probs=True,
        )

        if memory_state is None:
            memory_state = torch.zeros(
                (batch_size, self.num_heads, self.head_dim),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        memory_seq = memory_state.unsqueeze(1).expand(-1, seq_len, -1, -1)
        if seq_len > self.window_size:
            gate = self._compute_gate(hidden_states).unsqueeze(-1)
            evicted_v = v[:, : seq_len - self.window_size]
            update = self._project_memory_update(evicted_v)
            pad = torch.zeros(
                (batch_size, self.window_size, self.num_heads, self.head_dim),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            update = torch.cat([pad, update], dim=1)

            positions = torch.arange(seq_len, device=hidden_states.device)
            update_mask = positions >= self.window_size
            if self.mem_update_stride > 1:
                update_mask &= ((seqlen_offset + positions - self.window_size) % self.mem_update_stride) == 0
            update_mask = update_mask.view(1, seq_len, 1, 1)

            gate = torch.where(update_mask, gate, torch.ones_like(gate))
            update = torch.where(update_mask, update, torch.zeros_like(update))

            eps = self.mem_norm_eps
            gate_fp32 = gate.squeeze(-1).to(torch.float32)
            log_gate = torch.log(torch.clamp(gate_fp32, min=eps))
            if chunk_global_cumsum_scalar is None:
                log_prefix = torch.cumsum(log_gate, dim=1)
            else:
                log_prefix = chunk_global_cumsum_scalar(
                    log_gate,
                    head_first=False,
                    output_dtype=torch.float32,
                )
            prefix = torch.exp(log_prefix).clamp_min(eps).unsqueeze(-1)
            inv_prefix = 1.0 / prefix
            contrib = ((1.0 - gate.to(torch.float32)) * update.to(torch.float32)) * inv_prefix
            if chunk_global_cumsum_vector is None:
                accum = torch.cumsum(contrib, dim=1)
            else:
                accum = chunk_global_cumsum_vector(
                    contrib,
                    head_first=False,
                    output_dtype=torch.float32,
                )
            memory_seq = prefix * (memory_state.to(torch.float32).unsqueeze(1) + accum)
            memory_seq = memory_seq.to(q.dtype)

        memory_context = self._memory_context(
            memory_seq,
            seen_tokens=(seqlen_offset + torch.arange(1, seq_len + 1, device=hidden_states.device)).view(1, seq_len, 1, 1),
        )
        mem_k, mem_v = self._memory_kv(memory_context)
        mem_score = (q * mem_k).sum(-1).to(torch.float32) * self.scaling
        local_lse = local_lse.transpose(1, 2).to(torch.float32)
        total_lse = torch.logaddexp(local_lse, mem_score)
        local_weight = torch.exp(local_lse - total_lse).to(local_out.dtype).unsqueeze(-1)
        mem_weight = torch.exp(mem_score - total_lse).to(local_out.dtype).unsqueeze(-1)
        out = local_weight * local_out + mem_weight * mem_v.to(local_out.dtype)
        out = out.reshape(batch_size, seq_len, -1)
        out = self.o_proj(out)
        return out, memory_seq[:, -1], k, v

    def _forward_cached(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Any,
        *,
        seqlen_offset: int,
    ) -> tuple[torch.Tensor, Any]:
        batch_size = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(batch_size, 1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, 1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, 1, self.num_kv_heads, self.head_dim)

        q, k = self._apply_rope(q, k, seqlen_offset=seqlen_offset, max_seqlen=seqlen_offset + 1)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        memory_state, window_k, window_v = self._prepare_cache_state(
            past_key_values,
            batch_size,
            hidden_states.device,
            hidden_states.dtype,
        )
        if len(window_k) >= self.window_size:
            evicted_v = window_v.pop(0)
            window_k.pop(0)
            if not self.disable_memory and self._should_update_memory(seqlen_offset):
                memory_state = self._update_memory(memory_state, evicted_v, hidden_states[:, 0])

        window_k.append(k[:, 0])
        window_v.append(v[:, 0])

        k_flat, v_flat = self._pack_cache_state(window_k, window_v)
        past_key_values.update(
            recurrent_state=memory_state,
            attn_state=(k_flat, v_flat),
            layer_idx=self.layer_idx,
            offset=1,
            cache_kwargs=dict(window_size=self.window_size),
        )

        k_window = torch.stack(window_k, dim=2)
        v_window = torch.stack(window_v, dim=2)
        q_t = q[:, 0].unsqueeze(2)
        if self.disable_memory:
            k_cat = k_window
            v_cat = v_window
        else:
            mem_k, mem_v = self._memory_kv(self._memory_context(memory_state, seqlen_offset + 1))
            k_cat = torch.cat([mem_k.unsqueeze(2), k_window], dim=2)
            v_cat = torch.cat([mem_v.unsqueeze(2), v_window], dim=2)

        attn_out = F.scaled_dot_product_attention(q_t, k_cat, v_cat, is_causal=False)
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
        cache_len = 0
        if past_key_values is not None:
            cache_len = past_key_values.get_seq_length(self.layer_idx)
        if hidden_states.shape[1] > 1 and cache_len == 0:
            out, memory_state, k, v = self._forward_fast(
                hidden_states,
                attention_mask=attention_mask,
                seqlen_offset=0,
            )
            if use_cache and past_key_values is not None:
                window_k = [k[:, i] for i in range(max(0, k.shape[1] - self.window_size), k.shape[1])]
                window_v = [v[:, i] for i in range(max(0, v.shape[1] - self.window_size), v.shape[1])]
                k_flat, v_flat = self._pack_cache_state(window_k, window_v)
                past_key_values.update(
                    recurrent_state=memory_state,
                    attn_state=(k_flat, v_flat),
                    layer_idx=self.layer_idx,
                    offset=hidden_states.shape[1],
                    cache_kwargs=dict(window_size=self.window_size),
                )
            return out, None, past_key_values if use_cache else None
        if use_cache and past_key_values is not None and hidden_states.shape[1] == 1:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            out, past_key_values = self._forward_cached(hidden_states, past_key_values, seqlen_offset=seqlen_offset)
            return out, None, past_key_values

        memory_state, window_k, window_v = self._prepare_cache_state(
            past_key_values if use_cache else None,
            hidden_states.shape[0],
            hidden_states.device,
            hidden_states.dtype,
        )
        seqlen_offset = past_key_values.get_seq_length(self.layer_idx) if use_cache and past_key_values is not None else 0
        out, memory_state, window_k, window_v = self._forward_full(
            hidden_states,
            attention_mask=attention_mask,
            seqlen_offset=seqlen_offset,
            memory_state=memory_state,
            window_k=window_k,
            window_v=window_v,
        )

        if use_cache and past_key_values is not None:
            k_flat, v_flat = self._pack_cache_state(window_k, window_v)
            past_key_values.update(
                recurrent_state=memory_state,
                attn_state=(k_flat, v_flat),
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
    ) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
        if self.window_size is None:
            raise ValueError("inference_step requires window_size")

        batch_size = x_t.shape[0]
        if kv_cache is None:
            kv_cache = {"k": None, "v": None, "idx": 0, "filled": 0, "seen": 0}
        if memory_state is None:
            memory_state = torch.zeros((batch_size, self.num_heads, self.head_dim), device=x_t.device, dtype=x_t.dtype)

        q = self.q_proj(x_t).view(batch_size, 1, self.num_heads, self.head_dim)
        k = self.k_proj(x_t).view(batch_size, 1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x_t).view(batch_size, 1, self.num_kv_heads, self.head_dim)

        seqlen_offset = int(kv_cache.get("seen", 0))
        q, k = self._apply_rope(q, k, seqlen_offset=seqlen_offset, max_seqlen=seqlen_offset + 1)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        if kv_cache["k"] is None:
            kv_cache["k"] = torch.zeros(
                (batch_size, self.window_size, self.num_heads, self.head_dim),
                device=x_t.device,
                dtype=x_t.dtype,
            )
            kv_cache["v"] = torch.zeros_like(kv_cache["k"])

        if kv_cache["filled"] >= self.window_size:
            evicted_v = kv_cache["v"][:, kv_cache["idx"]]
            if not self.disable_memory and self._should_update_memory(seqlen_offset):
                memory_state = self._update_memory(memory_state, evicted_v, x_t[:, 0])

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
        if self.disable_memory:
            k_cat = k_seq
            v_cat = v_seq
        else:
            mem_k, mem_v = self._memory_kv(self._memory_context(memory_state, kv_cache["seen"]))
            k_cat = torch.cat([mem_k.unsqueeze(2), k_seq], dim=2)
            v_cat = torch.cat([mem_v.unsqueeze(2), v_seq], dim=2)
        attn_out = F.scaled_dot_product_attention(q_t, k_cat, v_cat, is_causal=False)
        attn_out = attn_out.squeeze(2).reshape(batch_size, 1, -1)
        return self.o_proj(attn_out), kv_cache, memory_state
