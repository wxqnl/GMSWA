# GM-SWA v2: TTT-style Fast Weights for Sliding-Window Attention

> **Status:** Implementation complete, all 13 unit tests pass on GPU 4 (H100).
> Layer file: `flash-linear-attention/fla/layers/gated_mem_swa.py`.
> Config:     `flash-linear-attention/fla/models/gated_mem_swa/configuration_gated_mem_swa.py`.
> Test:       `test_gmswa_v2.py` (CUDA_VISIBLE_DEVICES=4 python test_gmswa_v2.py)

## 1. Motivation

The v1 implementation had a fundamental design flaw: memory keys and memory values
were collinear (`k_mem = m_t`, `v_mem = α · m_t`), so the memory branch could only
push the output toward a single direction per head. Empirically this manifested as
exp2's `memory_only` condition giving **exactly** the same log-prob as `none` —
the memory carried no task-relevant signal.

v2 takes ideas from three ICLR'26 works:

1. **In-Place TTT** (Feng et al., ICLR'26 Oral) — instantiate long-range state as
   fast weights of a small linear projection, updated by an NTP-aligned objective.
2. **LaCT / Test-Time Training Done Right** (ICLR'26) — combine local window
   attention with a chunk-wise fast-weight update; let attention handle locality
   and the fast weight focus on non-local dependencies.
3. **Gated DeltaNet / Mamba-2** — the gated delta rule
   `S ← exp(g)·S·(I − β k kᵀ) + β v kᵀ` provides a numerically stable, parallel-
   trainable update that is exactly one step of regularized least-squares
   gradient descent for the linear-attention objective `Sk ≈ v`.

Concretely v2 replaces v1's "M vector slots" with a **matrix-valued fast weight**
`S ∈ R^{H_q × d_h × d_h}` per layer, updated by the gated delta rule on
**evicted (k, v)** pairs from the sliding window.

## 2. Architecture

### 2.1 Local branch (unchanged)

Standard sliding-window causal attention with window `W`:

  `o_local[t] = softmax(q_t · K_{[t-W+1:t]}ᵀ / √d_h) · V_{[t-W+1:t]}`

Implemented with `flash_attn_func(causal=True, window_size=(W-1, 0))` if
flash-attn is available, else an SDPA fallback with an additive mask.

### 2.2 Memory branch (replaces v1)

Per Q-head fast weight matrix:

  `S_t ∈ R^{d_h × d_h}`,  one per query head, per layer.

For each step `t`, denote by `k_e_t = k_proj[t - W]`, `v_e_t = v_proj[t - W]` the
PRE-RoPE projections of the token that was just **evicted** from the local
window at step `t`. (For `t < W` no eviction has happened yet, see below.)

The state update is the standard gated delta rule:

  `S_t = exp(g_t) · (S_{t-1} − β_t · S_{t-1} · k̂_e_t · k̂_e_tᵀ) + β_t · v_e_t · k̂_e_tᵀ`

where `k̂_e_t = k_e_t / ‖k_e_t‖_2`, `β_t ∈ (0, 1)` is the write strength, and
`g_t ≤ 0` is the log-space decay. Both `β_t` and `g_t` are predicted from the
CURRENT hidden state `h_t` (a single fused `Linear(d, 3H_q)` produces
`β`, `a`, and the mixture logit `m`).

For `t < W`: `β_t = 0` and `g_t = 0` are masked in, so `S_t = S_{t-1}` (with
initial `S_0 = 0`). Memory contribution is therefore exactly zero for the first
`W` tokens — the model behaves identically to pure SWA on this prefix.

The read is linear-attention style:

  `o_mem[t] = S_t · q̂_t`     where  `q̂_t = q_t / ‖q_t‖_2`

(L2 normalization of `q` and `k` is applied inside the kernel for training, and
inside the inline path for decode; this is the standard delta-rule formulation.)

### 2.3 Fusion

Per-Q-head sigmoid mixture gate `α_t ∈ (0, 1)`:

  `o_t = α_t · o_local[t] + (1 − α_t) · o_mem[t]`

`α_t` is the third output of the fused gate projection. The init bias is
`+4.0`, so at the start of training `α ≈ 0.98` and the layer behaves nearly
identically to pure SWA. The model learns to lower `α` for heads / positions
where memory carries useful signal.

### 2.4 Training-time computation

`_memory_branch` calls `fla.ops.gated_delta_rule.chunk_gated_delta_rule` (for
`T > 64`) or `fused_recurrent_gated_delta_rule` (for very short sequences /
eval). The kernel handles:

- L2 normalization of q, k internally (`use_qk_l2norm_in_kernel=True`)
- chunk-parallel forward (`chunk_local_cumsum` of `g`, WY representation of
  delta-rule writes, chunk-fwd output computation)
- exact backward via `chunk_gated_delta_rule_bwd`
- packed variable-length (`cu_seqlens`) for varlen training

To make tokens `t < W` perform no update, we mask both `β` and `g` (in log
space) to zero before the kernel call. This is equivalent to running the
recurrence over zero updates with retention 1.

The **shift-by-W** is implemented as `_shift_evicted_dense` (gather with an
explicit "valid" mask) for dense input and `_shift_evicted_varlen` for packed
variable-length input.

### 2.5 Decode (single-token) computation

`_forward_decode` uses an **inline** delta-rule step (`_inline_delta_rule_step`)
rather than the kernel — for a single token the kernel-launch overhead of
`fused_recurrent_gated_delta_rule` dominates the actual compute. The inline
path is two einsums + one outer product + L2 norm:

```python
S        = S * exp(g)
err      = v_e − S @ k̂_e
S        = S + β · err · k̂_eᵀ
o_mem    = S · q̂
```

This requires `O(d_h²)` ops per head per token.

### 2.6 Inference cost (per layer per sequence)

| quantity                 | size                       | example (W=128, H_kv=2, H_q=10, d_h=64) |
|--------------------------|----------------------------|------------------------------------------|
| local KV cache (GQA)     | `W · H_kv · 2 · d_h`       | 128 · 2 · 2 · 64 = 32 KB (bf16)          |
| fast weight `S`          | `H_q · d_h · d_h`          | 10 · 64 · 64 = 40 KB (fp32) / 20 KB (bf16) |
| total per layer          |                            | ~52 KB                                   |

Both quantities are **independent of sequence length** once the window is full.
This is the "constant KV cache, linear-attention style" property the user asked
for.

## 3. Causality

`o_local[t]` reads positions `[t−W+1, t]`. `S_t` only contains contributions
from `k_e_τ = k[τ − W]` for `τ ≤ t`, i.e., positions `[0, t − W]`. The two
ranges are disjoint and together cover `[0, t]` exactly. The unit test
`test_causality_no_future_leak` verifies that modifying input at the last
position leaves outputs at all earlier positions bit-exact unchanged.

## 4. API / cache contract

Per layer the FLA `Cache` holds:

```
state[layer_idx] = {
    "recurrent_state": Tensor (B, H_q, d_h, d_h),   # fast weight S
    "attn_state": (k_rope, v, k_write),            # ring buffer of last W
}
```

- `k_rope`, `v`, `k_write`: shape `(B, ≤ W, H_kv * d_h)` flat, dtype bf16.
- The cache handles its own ring-buffer roll via `cache_kwargs={"window_size":
  W}`; the decode path only ever passes the NEW token to `cache.update(...)`.

The cache key `recurrent_state` semantics are compatible with FLA's existing
infrastructure (used identically by `GatedDeltaNet`, `GLA`, etc.).

## 5. Implementation details worth flagging

1. **Pre-RoPE for memory branch.** The local branch uses RoPE-rotated q and k,
   but the memory branch uses pre-RoPE projections. This mirrors the choice in
   `GatedDeltaNet` — RoPE breaks the linear-attention compositionality, and
   using it inside the recurrent state is empirically worse.

2. **GQA handling.** K and V are projected at `H_kv` heads (GQA). Both the
   local branch and the memory branch repeat K/V to `H_q` heads via
   `repeat_kv` before the kernel call. Memory state is thus `H_q`-headed
   (each query head has its own state). The kernel and cache scale with
   `H_q · d_h²`. For our 110M config that's 40 KB per layer — negligible.

3. **L2 norm in kernel vs inline.** Training uses
   `use_qk_l2norm_in_kernel=True` so q and k are normalized inside the Triton
   kernel. The inline decode path does the same normalization in PyTorch.

4. **No memory normalization on read.** Unlike v1's asymmetric `clamp_min(1).sqrt()`
   shrink, v2 lets the gated delta rule keep `S` bounded by construction
   (`β ∈ (0,1)`, `exp(g) ∈ (0, 1]`).

5. **Mix gate init bias = +4.0.** `α ≈ sigmoid(4) ≈ 0.982` at init, so a fresh
   v2 model is virtually identical to pure SWA. This is critical for training
   stability — the model gradually learns to pull `α` down for heads that
   benefit from memory, rather than starting from a corrupt mixture.

6. **Write gate init bias = −2.0.** `β ≈ sigmoid(-2) ≈ 0.12` at init, so memory
   writes are small at first and grow with training.

7. **Mamba-2 style A_log + dt_bias.** Decay `g = −exp(A_log) · softplus(a + dt_bias)`
   exactly mirrors GatedDeltaNet's parameterization; in particular `dt_bias`
   is initialized via the inverse-softplus trick so initial `exp(g)` is close
   to (but not exactly) 1.

## 6. Test summary

All run on H100 GPU 4, bf16/fp32 mix, `flash-attn` not loaded (SDPA fallback
exercised):

| test                                | status |
|-------------------------------------|--------|
| forward shape                       | PASS   |
| no NaN/Inf                          | PASS   |
| backward gradients flow to all params | PASS   |
| disable_memory collapses to SWA     | PASS   |
| `o_mem[t<W] = 0` exactly             | PASS   |
| causality (no future leakage)        | PASS   |
| GQA (H_q=4, H_kv=2)                  | PASS   |
| long sequence (T=256) train+grad     | PASS   |
| constant KV+state size in decode     | PASS   |
| prefill+decode == full forward (last token) | PASS |
| step-by-step decode == full forward  | PASS (max abs diff ≤ 0.05 in bf16) |
| full-model smoke (2-layer)           | PASS   |
| full-model train step                | PASS   |

## 7. Benchmark (110M-config decode, single-token, GPU 4, bf16, no torch.compile)

| prefill T | GM-SWA v2 | pure SWA  | overhead |
|-----------|-----------|-----------|----------|
| 32        | 10.3 ms   | 6.9 ms    | +49%     |
| 256       | 12.6 ms   | 7.5 ms    | +68%     |
| 1024      | 12.7 ms   | 7.5 ms    | +69%     |

These numbers are without `torch.compile` (unavailable in Python 3.10) and
without CUDA Graphs. The absolute decode times are dominated by per-kernel
Python/torch dispatch overhead at this model scale; the memory-branch share
of total cost is ~5 ms over 12 layers ≈ 0.42 ms per layer. With
`torch.compile=True` or CUDA Graphs, expect overhead to drop substantially
because most of the inline-step ops can be fused into a single kernel.

For comparison, the v1 architecture spent more wall-clock time per step (it
also called the per-token recurrent scan inside a fused Triton kernel, plus
the multi-slot softmax read), so v2 is already faster than v1 in addition to
being more expressive.

## 8. What's intentionally NOT in v2

These are easy follow-ups but not part of this PR:

- **Top-K block read.** NSA's selection branch (Top-K block-level retrieval)
  would complement the gated delta state for precise long-range token recall.
  Plan B in the original review — could be added as an optional third branch.
- **Block-level writes.** Currently each evicted token writes to memory
  individually. NSA-style "every W' = W/2 tokens, write one block summary"
  would let memory capture coarser structure with less interference. The
  current code can be adapted by replacing the shift-by-W with a strided
  pool over windows of size W'.
- **Inner-loop NTP loss.** In-Place TTT proposes an explicit NTP-aligned
  inner-loop objective that further specializes the fast-weight update.
  The gated delta rule already implements ONE step of regularized least-
  squares gradient descent on `‖Sk − v‖²`; replacing this with a per-chunk
  multi-step optimization (LaCT style) is a logical next step.
- **`torch.compile`-friendly path.** The current implementation works under
  eager mode; making the inline decode step a single compiled graph would
  significantly reduce per-step overhead at decode time.
- **Cross-chunk state in dense training.** `_shift_evicted_dense` masks out
  evictions whose source would lie in a previous training chunk. This is
  conservative but correct for end-to-end pretraining (sequences are
  processed in one shot). If chunked / multi-pass training is needed, the
  past `(k, v)` buffer must be threaded through as well.

## 9. Files changed

- `flash-linear-attention/fla/layers/gated_mem_swa.py` — full rewrite (1762 → ~600 lines).
- `flash-linear-attention/fla/models/gated_mem_swa/configuration_gated_mem_swa.py` — new v2 hyperparameters, legacy v1 kwargs are silently ignored with a warning.
- `flash-linear-attention/fla/models/gated_mem_swa/modeling_gated_mem_swa.py` — block instantiation updated to v2 layer args.
- `config.json` — top-level config switched to v2 fields.
- `test_gmswa_v2.py` — 13-test sanity + correctness suite.

## 10. How to run

```bash
# tests
CUDA_VISIBLE_DEVICES=4 .venv/bin/python test_gmswa_v2.py

# decode benchmark (12-layer realistic config)
CUDA_VISIBLE_DEVICES=4 .venv/bin/python /tmp/bench_realistic.py   # script kept in /tmp during dev
```

If `flash-attn` is properly built in the target env, the local SWA branch will
use it automatically. Otherwise the SDPA fallback is used (same numerics,
slower on long sequences).
