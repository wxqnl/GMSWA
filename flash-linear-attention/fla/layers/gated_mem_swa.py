# GM-SWA v2: TTT-style fast weights for sliding-window attention.
# See paper/gmswa_v2_design.md.

from __future__ import annotations

import math
import os
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedRMSNormGated, RotaryEmbedding, ShortConvolution
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    _HAS_FLASH_ATTN = True
except Exception:
    flash_attn_func = None
    flash_attn_varlen_func = None
    _HAS_FLASH_ATTN = False


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, T, H_kv, D) -> (B, T, H_kv * n_rep, D)."""
    if n_rep == 1:
        return x
    B, T, H, D = x.shape
    return x.unsqueeze(3).expand(B, T, H, n_rep, D).reshape(B, T, H * n_rep, D)


def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(y))


def _is_fa_dtype(t: torch.Tensor) -> bool:
    return t.dtype in (torch.float16, torch.bfloat16)


class GatedMemSWA(nn.Module):
    """Gated Memory-augmented Sliding Window Attention (v2)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        rope_theta: float = 10000.0,
        max_position_embeddings: int | None = None,
        disable_memory: bool = False,
        disable_local: bool = False,
        mem_gate_logit_bias: float = -2.0,
        mix_gate_logit_bias: float = 4.0,
        a_log_init_lo: float = 1.0,
        a_log_init_hi: float = 16.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        mem_separate_proj: bool = False,
        mem_mode: str = "delta",
        mem_evicted_only: bool = True,
        mem_use_short_conv: bool = False,
        mem_conv_size: int = 4,
        mem_use_output_norm: bool = False,
        mem_swa_drop_prob: float = 0.0,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}")
        if window_size is None or window_size <= 0:
            raise ValueError("window_size must be a positive integer for GatedMemSWA")
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.window_size = int(window_size)
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.disable_memory = disable_memory
        # Memory-only ablation: drop the SWA branch from the OUTPUT (o = o_mem),
        # so the recurrent memory alone must predict. Used to test whether the
        # memory CAN learn sharp recall when not pre-empted by the window
        # (credit-assignment probe). The SWA branch is still computed to maintain
        # the K/V ring the memory reads at decode; only the mix is bypassed.
        self.disable_local = bool(disable_local) and not disable_memory
        self.mem_gate_logit_bias = float(mem_gate_logit_bias)
        self.mix_gate_logit_bias = float(mix_gate_logit_bias)
        self.a_log_init_lo = float(a_log_init_lo)
        self.a_log_init_hi = float(a_log_init_hi)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.dt_init_floor = float(dt_init_floor)

        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        self.memory_enabled = not disable_memory
        # When True, the memory branch gets its OWN content-retrieval q/k
        # projections (NoPE), decoupled from the positional SWA q/k. This lets
        # the memory learn addressing keys not bound to the local-attention
        # projections. Value (v) is shared: it is the content to copy out.
        self.mem_separate_proj = bool(mem_separate_proj) and self.memory_enabled
        # "delta" = gated-delta-rule recurrent readout (v3). "softmax" = exact
        # NoPE softmax retrieval over evicted (out-of-window) tokens (v4) — sharp
        # retrieval that recovers discrete recall the linear readout cannot.
        self.mem_mode = str(mem_mode)
        # Complementary design (default): the memory ingests ONLY evicted tokens
        # (those that left the window), so its constant state is spent on long-range
        # content the window cannot see. False = memory on ALL tokens (overlaps the
        # window) — the ablation for the complementary-decomposition claim.
        self.mem_evicted_only = bool(mem_evicted_only)
        # Stochastically drop the SWA branch during training (prob per forward) so
        # the memory must carry the full prediction alone -> forced to learn the
        # precise long-range recall the SWA shortcut otherwise hides. Inference: off.
        self.mem_swa_drop_prob = float(mem_swa_drop_prob)
        # `beta_proj` is exposed as None: the write-gate beta is produced by the
        # fused `gate_proj` (no separate Linear). Kept as an attribute so callers
        # can assert `layer.beta_proj is None` for disable_memory mode.
        self.beta_proj = None
        if self.memory_enabled:
            self.gate_proj = nn.Linear(dim, 3 * num_heads, bias=True)
            self.A_log = nn.Parameter(torch.empty(num_heads))
            self.dt_bias = nn.Parameter(torch.empty(num_heads))
        else:
            self.gate_proj = None
            self.A_log = None
            self.dt_bias = None
        if self.mem_separate_proj:
            self.mem_q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=qkv_bias)
            self.mem_k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=qkv_bias)
        else:
            self.mem_q_proj = None
            self.mem_k_proj = None

        # Optional short (causal depthwise) convolution on the memory's q/k/v —
        # the induction-head primitive that gated-delta recall relies on (cf.
        # GatedDeltaNet). Applied on the full sequence (pre-eviction) so the
        # convolved tokens are what the recurrent memory ingests. Opt-in: existing
        # checkpoints (flag absent -> False) are byte-for-byte unchanged.
        self.mem_use_short_conv = bool(mem_use_short_conv) and self.memory_enabled
        self.mem_conv_size = int(mem_conv_size)
        if self.mem_use_short_conv:
            self.mem_q_conv1d = ShortConvolution(
                hidden_size=num_heads * self.head_dim, kernel_size=mem_conv_size, activation="silu")
            self.mem_k_conv1d = ShortConvolution(
                hidden_size=num_kv_heads * self.head_dim, kernel_size=mem_conv_size, activation="silu")
            self.mem_v_conv1d = ShortConvolution(
                hidden_size=num_kv_heads * self.head_dim, kernel_size=mem_conv_size, activation="silu")
        else:
            self.mem_q_conv1d = self.mem_k_conv1d = self.mem_v_conv1d = None

        # Optional GDN-style gated RMSNorm on the memory delta-output before mixing.
        # GDN normalizes its delta readout with a per-head gated RMSNorm (o_norm);
        # without it the raw delta output is poorly scaled for precise retrieval.
        self.mem_use_output_norm = bool(mem_use_output_norm) and self.memory_enabled
        if self.mem_use_output_norm:
            self.mem_g_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
            self.mem_o_norm = FusedRMSNormGated(self.head_dim, eps=1e-6)
        else:
            self.mem_g_proj = None
            self.mem_o_norm = None

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=rope_theta)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not self.memory_enabled:
            return
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            bias = self.gate_proj.bias.view(3, self.num_heads)
            bias[0].fill_(self.mem_gate_logit_bias)
            bias[1].zero_()
            bias[2].fill_(self.mix_gate_logit_bias)
            self.A_log.uniform_(math.log(self.a_log_init_lo), math.log(self.a_log_init_hi))
            dt = torch.empty(self.num_heads).uniform_(self.dt_min, self.dt_max)
            dt = dt.clamp(min=self.dt_init_floor)
            self.dt_bias.copy_(_inverse_softplus(dt))

    # -------- gates --------

    def _compute_gates(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, T, hidden) -> (g, beta, mix_logit) all of shape (B, T, H_q)."""
        x = self.gate_proj(hidden_states)
        beta_logit, a_logit, mix_logit = x.chunk(3, dim=-1)
        beta = beta_logit.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a_logit.float() + self.dt_bias.float())
        return g, beta, mix_logit

    # -------- memory induction-head short conv --------

    def _apply_mem_conv(self, mem_q, mem_k, mem_v, conv_state_in, use_cache, cu_seqlens):
        """Causal depthwise short-conv on the memory q/k/v over the FULL sequence
        (pre-eviction), the induction primitive gated-delta recall relies on.
        Returns convolved (q, k, v) and the updated (cs_q, cs_k, cs_v) conv cache."""
        B, T = mem_q.shape[0], mem_q.shape[1]
        cs_q = cs_k = cs_v = None
        if conv_state_in is not None:
            cs_q, cs_k, cs_v = conv_state_in
        q, cs_q = self.mem_q_conv1d(
            mem_q.reshape(B, T, -1), cache=cs_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        k, cs_k = self.mem_k_conv1d(
            mem_k.reshape(B, T, -1), cache=cs_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        v, cs_v = self.mem_v_conv1d(
            mem_v.reshape(B, T, -1), cache=cs_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        mem_q = q.view(B, T, self.num_heads, self.head_dim)
        mem_k = k.view(B, T, self.num_kv_heads, self.head_dim)
        mem_v = v.view(B, T, self.num_kv_heads, self.head_dim)
        return mem_q, mem_k, mem_v, (cs_q, cs_k, cs_v)

    # -------- evicted-token shift (full prefill / training) --------

    def _shift_evicted_dense(
        self,
        k_pre: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, H, D = k_pre.shape
        W = self.window_size
        valid = torch.zeros(B, T, dtype=torch.float32, device=k_pre.device)
        if T <= W:
            return torch.zeros_like(k_pre), torch.zeros_like(v), valid
        pad_k = k_pre.new_zeros(B, W, H, D)
        pad_v = v.new_zeros(B, W, H, D)
        k_e = torch.cat([pad_k, k_pre[:, : T - W]], dim=1)
        v_e = torch.cat([pad_v, v[:, : T - W]], dim=1)
        valid[:, W:] = 1.0
        return k_e, v_e, valid

    def _shift_evicted_varlen(
        self,
        k_pre: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, total_T, H, D = k_pre.shape
        assert B == 1, "varlen path expects batch dim 1"
        W = self.window_size
        device = k_pre.device
        cu = cu_seqlens.to(device)
        pos = torch.arange(total_T, device=device)
        seq_id = torch.searchsorted(cu[1:].contiguous(), pos, right=True)
        bos = cu[seq_id]
        offset = pos - bos
        valid = (offset >= W).to(torch.float32).unsqueeze(0)
        src_idx = (pos - W).clamp_min(0)
        k_e = k_pre.index_select(1, src_idx)
        v_e = v.index_select(1, src_idx)
        mask = valid.view(1, total_T, 1, 1)
        k_e = k_e * mask.to(k_e.dtype)
        v_e = v_e * mask.to(v_e.dtype)
        return k_e, v_e, valid

    # -------- memory branch (prefill / training) --------

    def _memory_branch(
        self,
        q_pre: torch.Tensor,
        k_pre: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        force_recurrent: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the gated-delta-rule memory branch on a full prefill segment.

        Args:
            q_pre: (B, T, H_q, D) pre-RoPE queries.
            k_pre, v: (B, T, H_kv, D) pre-RoPE keys / values.
            hidden_states: (B, T, dim) used to predict g and beta.
            initial_state: (N, H_q, D, D) or None.
            output_final_state: pass-through to the kernel.
            cu_seqlens: varlen boundaries or None.
            force_recurrent: prefer fused_recurrent kernel (single-token decode etc.).
        """
        g, beta, _ = self._compute_gates(hidden_states)

        if not self.mem_evicted_only:
            # ablation: memory ingests ALL tokens (overlaps the window)
            k_e, v_e = k_pre, v
            valid = torch.ones(k_pre.shape[0], k_pre.shape[1], device=k_pre.device, dtype=torch.float32)
        elif cu_seqlens is not None:
            k_e, v_e, valid = self._shift_evicted_varlen(k_pre, v, cu_seqlens)
        else:
            k_e, v_e, valid = self._shift_evicted_dense(k_pre, v)

        valid_h = valid.unsqueeze(-1)
        g_eff = g * valid_h
        beta_eff = beta * valid_h.to(beta.dtype)

        k_e_q = _repeat_kv(k_e, self.num_kv_groups)
        v_e_q = _repeat_kv(v_e, self.num_kv_groups)

        use_recurrent = force_recurrent or (q_pre.shape[1] <= 64 and not self.training)
        op = fused_recurrent_gated_delta_rule if use_recurrent else chunk_gated_delta_rule
        o_mem, final_state = op(
            q=q_pre,
            k=k_e_q,
            v=v_e_q,
            g=g_eff,
            beta=beta_eff,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        return o_mem, final_state

    # -------- local SWA branch --------

    def _local_attention(
        self,
        q_rope: torch.Tensor,
        k_rope: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.LongTensor | None,
        is_cached_decode: bool,
    ) -> torch.Tensor:
        B, T_q, H_q, D = q_rope.shape
        T_kv = k_rope.shape[1]
        H_kv = k_rope.shape[2]
        W = self.window_size
        use_flash = _HAS_FLASH_ATTN and _is_fa_dtype(q_rope)

        if use_flash and cu_seqlens is None and not is_cached_decode:
            return flash_attn_func(q_rope, k_rope, v, causal=True, window_size=(W - 1, 0))
        if use_flash and cu_seqlens is not None and not is_cached_decode:
            q = q_rope.squeeze(0)
            k = k_rope.squeeze(0)
            v_ = v.squeeze(0)
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            return flash_attn_varlen_func(
                q, k, v_,
                cu_seqlens_q=cu_seqlens.to(torch.int32),
                cu_seqlens_k=cu_seqlens.to(torch.int32),
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(W - 1, 0),
            ).unsqueeze(0)
        if use_flash and is_cached_decode:
            # Cached decode: q has T_q new tokens against a ring of T_kv. flash_attn
            # only allows windowing with `(W-1, 0)` and full causal — here T_kv may
            # be smaller than the absolute position, so the window meaning aligns
            # with "attend to all of k" plus causal. We just request causal SWA
            # with q and k differing in seq length.
            return flash_attn_func(q_rope, k_rope, v, causal=True, window_size=(W - 1, 0))

        # SDPA fallback.
        q = q_rope.transpose(1, 2)
        k = k_rope.transpose(1, 2)
        v_ = v.transpose(1, 2)
        if H_kv != H_q:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_ = v_.repeat_interleave(self.num_kv_groups, dim=1)

        if cu_seqlens is not None:
            outs = []
            for i in range(len(cu_seqlens) - 1):
                s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
                q_i, k_i, v_i = q[:, :, s:e], k[:, :, s:e], v_[:, :, s:e]
                outs.append(self._sdpa_window(q_i, k_i, v_i, W))
            return torch.cat(outs, dim=2).transpose(1, 2)

        if is_cached_decode:
            q_pos = T_kv - T_q + torch.arange(T_q, device=q.device)
            k_pos = torch.arange(T_kv, device=q.device)
            diff = q_pos[:, None] - k_pos[None, :]
            mask = (diff >= 0) & (diff < W)
            o = F.scaled_dot_product_attention(q, k, v_, attn_mask=mask, is_causal=False, scale=self.scaling)
            return o.transpose(1, 2)

        return self._sdpa_window(q, k, v_, W).transpose(1, 2)

    def _sdpa_window(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, W: int) -> torch.Tensor:
        T = q.shape[2]
        idx = torch.arange(T, device=q.device)
        diff = idx[:, None] - idx[None, :]
        mask = (diff >= 0) & (diff < W)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, scale=self.scaling)

    # -------- cached-decode memory step --------

    def _cached_memory_step(
        self,
        q_pre: torch.Tensor,
        k_pre_new: torch.Tensor,
        v_new: torch.Tensor,
        hidden_states: torch.Tensor,
        old_k_pre_ring: torch.Tensor | None,
        old_v_ring: torch.Tensor | None,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One delta-rule pass over the T new tokens, reading evicted (k, v)
        from the pre-update ring buffer.
        """
        B, T, H_q, D = q_pre.shape
        H_kv = k_pre_new.shape[2]
        W = self.window_size
        g, beta, _ = self._compute_gates(hidden_states)

        if old_k_pre_ring is None:
            combined_k = k_pre_new
            combined_v = v_new
            len_old = 0
        else:
            combined_k = torch.cat([old_k_pre_ring, k_pre_new], dim=1)
            combined_v = torch.cat([old_v_ring, v_new], dim=1)
            len_old = old_k_pre_ring.shape[1]

        # For each new token at new-index t (absolute pos seqlen_offset + t), the
        # evicted absolute pos is seqlen_offset + t - W, which lives at
        # combined-index (len_old - W + t).
        idx = torch.arange(T, device=q_pre.device) + (len_old - W)
        valid = (idx >= 0).to(torch.float32).view(1, T).expand(B, T)
        idx_clamped = idx.clamp(min=0, max=combined_k.shape[1] - 1)
        k_e = combined_k.index_select(1, idx_clamped)
        v_e = combined_v.index_select(1, idx_clamped)
        mask = valid.view(B, T, 1, 1).to(k_e.dtype)
        k_e = k_e * mask
        v_e = v_e * mask
        valid_h = valid.unsqueeze(-1)
        g_eff = g * valid_h
        beta_eff = beta * valid_h.to(beta.dtype)
        k_e_q = _repeat_kv(k_e, self.num_kv_groups)
        v_e_q = _repeat_kv(v_e, self.num_kv_groups)
        o_mem, final_state = fused_recurrent_gated_delta_rule(
            q=q_pre,
            k=k_e_q,
            v=v_e_q,
            g=g_eff,
            beta=beta_eff,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )
        return o_mem, final_state

    # -------- forward --------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: "Cache | None" = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, "Cache | None"]:
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "GatedMemSWA expects a 2D attention_mask of shape (B, T) for padding."
            )
        cu_seqlens = kwargs.get("cu_seqlens")
        B, T, _ = hidden_states.shape

        q_pre = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        k_pre = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)

        # Memory branch read/write keys. With separate projections the memory
        # learns its own (NoPE) content-addressing q/k, decoupled from SWA.
        if self.mem_separate_proj:
            mem_q = self.mem_q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
            mem_k = self.mem_k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        else:
            mem_q, mem_k = q_pre, k_pre
        mem_v = v  # value the memory ingests (kept distinct from the SWA value)

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # Optional induction-head short conv on the memory q/k/v (full-sequence,
        # causal, pre-eviction). conv cache is threaded for single-token decode.
        conv_state_out = None
        if self.mem_use_short_conv:
            conv_state_in = last_state.get("conv_state") if last_state is not None else None
            mem_q, mem_k, mem_v, conv_state_out = self._apply_mem_conv(
                mem_q, mem_k, mem_v, conv_state_in, use_cache, cu_seqlens,
            )

        # Absolute seq position (tokens seen so far). Falls back to ring length.
        seqlen_offset = 0
        if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
            seqlen_offset = int(past_key_values.get_seq_length(self.layer_idx))
        elif last_state is not None and last_state.get("attn_state") is not None:
            seqlen_offset = int(last_state["attn_state"][0].shape[1])

        old_k_pre_ring = None
        old_v_ring = None
        if last_state is not None and last_state.get("attn_state") is not None:
            attn_state_old = last_state["attn_state"]
            len_old = attn_state_old[0].shape[1]
            old_k_pre_ring = attn_state_old[2].view(B, len_old, self.num_kv_heads, self.head_dim)
            old_v_ring = attn_state_old[3].view(B, len_old, self.num_kv_heads, self.head_dim)

        max_seqlen = T + seqlen_offset
        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q_rope, k_rope = self.rotary(
            q_pre, k_pre,
            seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens,
        )

        is_cached_decode = use_cache and seqlen_offset > 0

        if past_key_values is not None and use_cache:
            new_ring = past_key_values.update(
                attn_state=(
                    k_rope.flatten(-2, -1),
                    v.flatten(-2, -1),
                    mem_k.flatten(-2, -1),
                    mem_v.flatten(-2, -1),
                ),
                conv_state=conv_state_out,
                layer_idx=self.layer_idx,
                offset=T,
                cache_kwargs=dict(window_size=self.window_size),
            )["attn_state"]
            if is_cached_decode:
                # Single-token (or post-prefill) decode: the windowed ring holds
                # exactly the keys the new query may attend to.
                k_rope_ring = new_ring[0].view(B, -1, self.num_kv_heads, self.head_dim)
                v_for_local = new_ring[1].view(B, -1, self.num_kv_heads, self.head_dim)
            else:
                # Multi-token prefill: attend over the FULL current segment; the
                # sliding-window mask in `_local_attention` enforces SWA. Using the
                # cache's re-windowed ring here would misalign causal attention for
                # every non-last query and corrupt the stored K/V that deeper
                # layers (and later decode steps) read back. The windowed ring is
                # still stored above for subsequent single-token decode.
                k_rope_ring = k_rope
                v_for_local = v
        else:
            k_rope_ring = k_rope
            v_for_local = v

        o_local = self._local_attention(
            q_rope=q_rope, k_rope=k_rope_ring, v=v_for_local,
            cu_seqlens=cu_seqlens, is_cached_decode=is_cached_decode,
        )

        if not self.memory_enabled:
            o = o_local
            recurrent_state_out = None
            mix_logit = None
        else:
            initial_state = None
            if last_state is not None:
                initial_state = last_state.get("recurrent_state")

            if is_cached_decode:
                o_mem, recurrent_state_out = self._cached_memory_step(
                    q_pre=mem_q, k_pre_new=mem_k, v_new=mem_v,
                    hidden_states=hidden_states,
                    old_k_pre_ring=old_k_pre_ring, old_v_ring=old_v_ring,
                    initial_state=initial_state, output_final_state=use_cache,
                )
            else:
                o_mem, recurrent_state_out = self._memory_branch(
                    q_pre=mem_q, k_pre=mem_k, v=mem_v,
                    hidden_states=hidden_states,
                    initial_state=initial_state, output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                )

            # GDN-style gated RMSNorm on the memory output (shapes it for precise
            # retrieval) before blending with the SWA branch.
            if self.mem_use_output_norm:
                g_out = self.mem_g_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
                o_mem = self.mem_o_norm(o_mem, g_out)

            if self.disable_local:
                # memory-only ablation: bypass the SWA mix; the memory alone predicts.
                o = o_mem.to(o_local.dtype)
                mix_logit = None
            else:
                _, _, mix_logit = self._compute_gates(hidden_states)
                alpha = mix_logit.sigmoid().unsqueeze(-1)
                # DIAGNOSTIC ONLY: force the mix to a constant at inference
                # (GMSWA_FORCE_ALPHA=0.0 -> memory-only). Unset -> learned gate.
                _force = os.environ.get("GMSWA_FORCE_ALPHA")
                if _force is not None:
                    alpha = torch.full_like(alpha, float(_force))
                # SWA-drop prob: runtime (annealed curriculum) if set, else static.
                drop_p = getattr(self, "_runtime_drop_p", None)
                if drop_p is None:
                    drop_p = self.mem_swa_drop_prob
                if self.training and drop_p > 0.0:
                    # Branchless + on-device rand => activation-checkpoint safe.
                    keep = (torch.rand((), device=o_local.device) >= drop_p).to(alpha.dtype)
                    alpha = alpha * keep
                o = alpha * o_local + (1.0 - alpha) * o_mem.to(o_local.dtype)

        if past_key_values is not None and use_cache and recurrent_state_out is not None:
            past_key_values.update(
                recurrent_state=recurrent_state_out,
                layer_idx=self.layer_idx,
                offset=0,
            )

        out = self.o_proj(o.reshape(B, T, -1))
        return out, None, past_key_values
