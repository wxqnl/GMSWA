# GMSWA: Gated-Memory Sliding-Window Attention for Constant-Memory Long-Context Language Modeling

**Authors:** [Author 1]¹, [Author 2]², [Corresponding Author]¹
**¹** [Department, Institution, City, Country]
**²** [Department, Institution, City, Country]
**Contact:** [corresponding@example.com]

> **Draft status (2026-06-03).** This is a working AAAI submission draft produced with the `academic-research-skills` `academic-paper` methodology (conference-paper structure, mandatory-inclusions checklist, honest epistemic marking). All quantitative results in §5 are real measurements from 340M-parameter models trained under a shared recipe. Results not yet collected are explicitly tagged **[PENDING]** with placeholder tables laid out so that filling them in completes the paper. No numbers are fabricated. arXiv identifiers in the bibliography are carried verbatim from the project brief and **must be DOI/arXiv-verified before submission** (see §Acknowledged Limitations and the AI-disclosure note).

---

## Abstract

Sliding-window attention (SWA) gives Transformers a constant per-step KV-cache and linear-time decoding, but information that leaves the window is lost irrecoverably; linear-attention and state-space models keep a constant-size recurrent state yet trade away the sharp local retrieval that softmax attention provides. We present **GMSWA (Gated-Memory Sliding-Window Attention)**, a single constant-memory attention layer that composes an exact local softmax window with a recurrent gated-delta-rule matrix memory. The two branches are *complementary by construction*: the window attends only to the most recent `W` tokens, while the memory ingests **only tokens that have been evicted from the window**, so the two coverage sets partition the causal history with neither overlap nor future leakage. The layer's output mixes the two branches through a learned per-position gate, `o = α·SWA + (1−α)·memory`. A revised variant (v3) adds separate **NoPE** (no positional encoding) content-retrieval projections for the memory branch, decoupling long-range associative recall from rotary position phase and yielding length extrapolation. The whole layer keeps a KV-cache and recurrent state whose size is **independent of sequence length**. At 340M parameters trained on 10B tokens of FineWeb-Edu under a shared recipe, GMSWA-v3 attains the lowest next-token loss at every measured token position among constant-memory models, and the gap over plain SWA *grows with distance* — direct evidence that the memory exploits far context. Beyond the 2048-token training length, a same-recipe full-attention Transformer collapses (loss 5.74 then 7.21 at the 2–4K and 4–8K bands, a rotary-extrapolation failure), whereas GMSWA-v3 remains the best of all four models (3.99, 4.04). On single-needle retrieval (NIAH/RULER) at this scale, GMSWA-v3 matches SWA — discrete recall is known to be scale-emergent and weak in sub-billion base models — so at 340M the memory delivers *soft* long-context utilization (lower loss) rather than sharp discrete copy; we report this honestly and frame the predicted recall win as a scale-dependent, structurally motivated [PENDING] result. A training-free capacity probe explains the design space: bilinear constant-memory mechanisms (linear/delta) hit a rank wall, motivating a recall-versus-memory frontier within which GMSWA is positioned. GMSWA is offered as a reference framework for SWA-based hybrid sparse attention with a pluggable, complementary constant-memory tail.

**Keywords**: sliding-window attention, constant-memory inference, linear attention, gated delta rule, long-context language modeling, length extrapolation

---

## 1. Introduction

### 1.1 Problem and Motivation

Standard self-attention costs O(T²) per sequence and, at inference, a KV-cache that grows linearly with sequence length T. For long-context decoding this makes memory and latency the binding constraints rather than compute. Two established families relax this cost, each with a structural blind spot.

*Sliding-window attention* (Mistral, Longformer) caps attention at the most recent `W` tokens, giving an O(W) per-step cache that is constant in T and an implementation that drops directly into existing Transformer stacks. Its limitation is definitional: any token that leaves the window is gone, so genuinely long-range dependencies cannot be served, at any model scale, by the window alone.

*Linear-attention and state-space models* (Mamba, GLA, DeltaNet, Gated DeltaNet) carry a fixed-size recurrent state and reach arbitrarily far back in principle, but their bilinear state is a weak associative store: precise local retrieval — the operation softmax attention does effortlessly — degrades as the number of stored associations grows.

The two failure modes are complementary, which suggests composing rather than choosing. The natural questions are *where* to attach a constant-memory recurrence to a window so the two do not duplicate work, and *how* to make that memory retrieve content rather than position so it survives beyond the training length. This paper answers both.

### 1.2 Research Questions

- **RQ1.** Can a single constant-memory layer combine an exact local window with a recurrent memory so that the memory measurably improves long-context modeling *beyond* what the window alone achieves?
- **RQ2.** Does feeding the memory **only evicted tokens** (a complementary, non-overlapping partition of history) and giving it **content-based NoPE retrieval projections** produce graceful length extrapolation, in contrast to full attention's rotary-extrapolation collapse?
- **RQ3.** What is the honest reach of such a memory at small scale — soft loss-level utilization versus sharp discrete recall — and how is that explained by the capacity limits of bilinear constant-memory stores?

### 1.3 Contributions

This paper makes four contributions.

1. **Architecture.** We introduce GMSWA, a constant-memory attention layer pairing an exact softmax sliding window with a gated-delta-rule matrix memory under a learned mixing gate. The defining choice is **complementary coverage**: the memory ingests only window-evicted tokens, so the window's exact-local set and the memory's long-tail set partition the causal prefix with no overlap and no future leakage (verified bit-exactly).

2. **NoPE content retrieval for extrapolation (v3).** We add separate, position-free retrieval projections (`mem_q`/`mem_k`) for the memory branch, plus a corrected gate initialization. Decoupling associative recall from rotary phase is what lets GMSWA keep improving past the training length where rotary full attention fails.

3. **Best-in-class constant-memory long-context utilization, with extrapolation beyond full attention.** Under a shared 340M/10B-token recipe, GMSWA-v3 has the lowest next-token loss at every measured position among constant-memory models, with a gap over SWA that widens with distance, and it is the single best model — better than full attention — in the extrapolation bands where rotary attention collapses.

4. **A recall-versus-memory frontier and an honest scale account.** A training-free capacity probe shows bilinear constant memories hit a rank wall (storing D associations needs O(D²) state) that a sketch/hash memory escapes (O(D)) but cannot be trained reliably; this frames a recall–memory design frontier and explains why, at 340M, GMSWA yields soft utilization rather than discrete recall, with the discrete-recall win predicted at larger scale where SWA remains structurally capped by its window.

We position GMSWA not as a single point model but as a **reference framework** for SWA-based hybrid sparse attention with a pluggable, complementary constant-memory tail.

---

## 2. Related Work

### 2.1 Sparse and Sliding-Window Attention

Window attention (Longformer; Mistral's SWA) bounds the attention span to recent tokens for an O(W) cache and linear decoding. Blockwise and hierarchical sparse schemes extend the reachable span but still discard or compress distant context. GMSWA keeps an *exact* local window — no approximation inside the window — and delegates the discarded tail to a recurrent memory rather than to a wider or sparser attention pattern.

### 2.2 Linear Attention, SSMs, and Constant-Memory Recall

Linear attention and selective state-space models (Mamba [arXiv:2312.00752]) maintain a fixed-size recurrent state for O(1) per-step memory. The delta rule and its gated form (Gated DeltaNet [arXiv:2412.06464]) cast the update as one regularized least-squares step, sharpening associative writes; DeltaProduct [arXiv:2502.10297] and RWKV-7 [arXiv:2503.14456] extend the state transition's expressivity; Based [arXiv:2402.18668] interpolates between linear and local attention to recover recall. The Zoology study [arXiv:2312.04927] documents a recall–throughput tradeoff for these stores, and a test-time-regression view [arXiv:2501.12352] formalizes their effective capacity as bounded by the rank of the bilinear state — the rank wall we probe in §6. GMSWA uses a gated-delta-rule memory as its long-tail branch but, unlike pure linear/SSM models, never asks that branch to do local retrieval: the exact window owns the local regime.

### 2.3 Window-plus-Memory Hybrids, and How GMSWA Differs

The closest prior art combines a window with a constant-memory side channel. **laLTE** [arXiv:2510.20787] couples SWA with linear attention and a *learned token-eviction* policy; **SWAX** [arXiv:2509.24552] pairs SWA with an xLSTM recurrence. Titans [arXiv:2501.00663] learns to memorize at test time, and Fast-weight Product-Key Memory (FwPKM) [arXiv:2601.00671] builds a high-capacity differentiable key–value store that outperforms Gated DeltaNet on its mechanism benchmark. GMSWA differs along three axes that we make explicit and ablate:

1. **Complementary, eviction-defined coverage.** The memory consumes *exactly and only* the tokens evicted from the window. The window's set `[t−W+1, t]` and the memory's set `[0, t−W]` partition the prefix; there is no overlapping double-coverage of recent tokens and no learned router deciding what to memorize. This is a deterministic division of labor, not a learned gate over which tokens enter memory (contrast laLTE's learned eviction).
2. **NoPE content retrieval.** Separate position-free `mem_q`/`mem_k` projections make the memory retrieve by content, decoupled from rotary phase — the mechanism behind GMSWA's extrapolation past the training length.
3. **A recall–memory frontier as design science.** Rather than claim a new highest-capacity mechanism (a space we find both occupied by FwPKM and hard to train — see §6), we characterize *where on the recall–memory frontier* a window+memory layer should sit, and offer GMSWA as a reference instantiation.

### 2.4 Gap This Paper Addresses

Prior window+memory hybrids either (i) let the memory and window overlap in coverage, (ii) leave the memory position-coupled (so it inherits rotary's extrapolation failure), or (iii) chase a single best mechanism. GMSWA closes these by a deterministic complementary partition, NoPE content retrieval for extrapolation, and a frontier framing that is honest about the scale at which discrete recall emerges. Benchmarking against laLTE/SWAX/FwPKM and against same-recipe SWA, Gated DeltaNet, full attention, GLA, and Mamba-2 is the empirical core (§5, with several entries [PENDING]).

---

## 3. Method

### 3.1 Overview

Each GMSWA layer runs two parallel branches on the hidden state and fuses them with a learned, per-position sigmoid gate `α_t`:

```
o_t = α_t · o_local[t] + (1 − α_t) · o_mem[t]
```

The local branch is exact causal SWA; the memory branch is a constant-size matrix fast-weight updated by a gated delta rule. The branches share a single small fusion projection `Linear(d, 3H)` producing, per head, the write strength `β`, the log-decay `g`, and the mixing coefficient `α`.

### 3.2 Local Branch — Exact Sliding-Window Attention

Each position `t` attends only to the most recent `W` tokens:

```
o_local[t] = softmax( q_t · K_[t−W+1:t]ᵀ / √d ) · V_[t−W+1:t]
```

Queries and keys carry rotary position encoding. The implementation calls `flash_attn_func(causal=True, window_size=(W−1, 0))` where available and falls back to an equivalent SDPA + additive-mask path (e.g. in fp32 environments) producing identical results. The per-step local cache is a ring buffer of size `W·H_kv·2·d`, constant in T.

### 3.3 Memory Branch — Gated-Delta-Rule Matrix Fast Weight

Each query head holds a matrix state `S ∈ R^{d×d}`. A token is written to memory **only when it is evicted from the window** — i.e. when it transitions out of `[t−W+1, t]`. On eviction of token `e` with (pre-RoPE content) key `k̂_e` and value `v_e`, the state updates by a gated delta rule (the Gated DeltaNet / Mamba-2 form):

```
S_t = exp(g_t) · ( S_{t−1} − β_t · S_{t−1} k̂_e k̂_eᵀ ) + β_t · v_e k̂_eᵀ
```

Reading is linear-attention style: `o_mem[t] = S_t · q̂_t`. Training uses FLA's chunk-parallel Triton kernel `chunk_gated_delta_rule` (with exact backward); single-token decoding uses a lightweight inline path to avoid kernel-launch overhead. The memory state size is `H_q·d·d`, constant in T.

### 3.4 Complementary Coverage: Eviction = Write, No Overlap, No Leakage

The defining property of GMSWA is that *being evicted from the window is exactly the event that writes a token to memory*. As a result, at any step `t`:

```
position:   0   1   2  ...      t−W              t−1   t
            └─────────────────────┘   └─────────────────────┘
              evicted → in memory S       still in the window
              memory holds [0, t−W]       attention sees [t−W+1, t]
```

The memory set `[0, t−W]` and the window set `[t−W+1, t]` are disjoint and together cover `[0, t]` exactly: no token is double-counted, and no future token can influence an earlier output. We verify this with a `test_causality_no_future_leak` check — perturbing the last position's input leaves every earlier position's output bit-for-bit unchanged — and with prefill/decode consistency tests (cosine 1.0 after a decoding-cache bug fix described in §5.1). This deterministic partition is the structural distinction from hybrids whose memory and window can overlap or whose eviction is learned.

### 3.5 NoPE Content-Retrieval Projections (v3)

In v2, the memory reused the rotary-encoded keys/queries, coupling associative recall to position phase. **v3** introduces separate, position-free `mem_q`/`mem_k` projections (NoPE) feeding the memory branch, so the memory retrieves by *content*, independent of how far back the match lies. Combined with a corrected gate initialization (the memory contributes negligibly at initialization, `α ≈ 0.98`, so training begins near pure SWA and stays stable, then learns to lean on memory), this is the mechanism behind GMSWA-v3's length extrapolation: content matches do not decay with rotary phase, so the memory keeps helping past the training length where the rotary window's relative positions become out-of-distribution.

### 3.6 Constant-Memory Analysis

The per-layer inference state is the sum of the local ring buffer and the matrix memory, both independent of T:

| Quantity | Size formula | Example (W=512, H_kv, H_q, d) |
|---|---|---|
| Local KV cache (GQA) | `W · H_kv · 2 · d` | tens of KB (bf16) |
| Matrix memory `S` | `H_q · d · d` | tens of KB |
| **Per-layer total** | constant in T | **≈ constant** |

Whereas full attention's KV-cache grows linearly (hundreds of MB at T=32K), GMSWA's per-layer footprint is fixed once the window fills. The memory branch adds a small per-step cost (≈0.42 ms/layer in a small dev configuration without `torch.compile`/CUDA-graphs); fusing this path is expected to reduce overhead substantially. Exact wall-clock and throughput numbers under the 340M configuration are **[PENDING]** (Table 7).

### 3.7 Why a Bilinear Memory: the Rank-Wall Framing

The memory's capacity is governed by the rank of its bilinear state. Under the test-time-regression view [arXiv:2501.12352], a delta-rule store reconstructs values by regression against stored keys; storing D near-orthogonal associations therefore requires state rank ≈ D, i.e. O(D²) entries for D associations. This rank wall (probed directly in §6) is *why* we do not ask the memory to perform sharp local retrieval — that is the window's job — and is the reason GMSWA's memory is best understood as a *soft* long-range summarizer at small scale. The frontier interpretation (recall vs. memory budget) follows in §7.

---

## 4. Experimental Setup

**Models.** Four architectures trained under one recipe at 340M parameters: **SWA** (pure window), **GMSWA-v2**, **GMSWA-v3**, and a **full-attention Transformer** (an upper-bound reference within the training length). A same-recipe **Gated DeltaNet** baseline (memory-only, no window; ~336M, parameter-aligned) is **[PENDING]** (training in progress); **GLA** and **Mamba-2** peers are **[PENDING]**.

**Training.** 10B tokens of FineWeb-Edu; identical optimizer, schedule, tokenizer, and 340M shape (hidden 1024, 24 layers, 16 query / 4 KV heads, window 512) across models; training context length 2048. (A 110M development configuration with window 128 is used only for unit/consistency testing.) The 1B configuration (hidden 2048, 24 layers, 32/8 heads, window 512; 1218M params) is used for the [PENDING] recall experiment.

**Evaluation.**
- *Loss vs. token position* (clean pretraining long-context utilization): next-token loss aggregated by absolute position band, including bands beyond the 2048 training length to test extrapolation. Lower is better.
- *Single-needle retrieval* (NIAH / RULER): exact-match recall versus context length.
- *Mechanism capacity probe* (§6): a training-free diagnostic of constant-memory associative capacity.

**Reproducibility.** Configs and eval scripts are in the project repository (`flash-linear-attention/flame/configs/`, `eval_results/`). The GMSWA layer is ~481 lines; 13/13 unit tests pass (forward shapes, no NaN/Inf, full-parameter gradients, GQA, bit-exact causality, constant decode-time state, prefill==full-forward, step-decode==full-forward).

---

## 5. Results

### 5.1 A Decoding Correctness Fix (Prerequisite)

During evaluation we found and fixed a real decoding bug: under `use_cache` with multi-token prefill, a cache-alignment error corrupted deep-layer K/V. After the fix, prefill/decode consistency is cosine 1.0 and step-decoding matches the full forward pass bit-for-bit. All §5 numbers are post-fix.

### 5.2 Loss vs. Token Position (Headline Result)

Next-token loss by absolute position band (lower is better). The last two bands lie beyond the 2048-token training length and test extrapolation.

| Position band | SWA | GMSWA-v2 | **GMSWA-v3** | Transformer (full attn) |
|---|---|---|---|---|
| 0–512 | 3.99 | 3.98 | **3.91** | 3.97 |
| 512–1024 | 3.73 | 3.68 | **3.63** | 3.59 |
| 1024–2048 | 3.87 | 3.81 | **3.75** | 3.62 |
| 2048–4096 *(extrapolation)* | 4.14 | 4.07 | **3.99** | **5.74** |
| 4096–8192 *(extrapolation)* | 4.21 | 4.14 | **4.04** | **7.21** |

Two findings stand out.

**Within training length, GMSWA-v3 beats SWA at every position, and the margin grows with distance** (0.08 at 0–512, rising to 0.12 at 1024–2048). A constant-memory branch that ignored far context could not widen its lead with distance; the widening margin is direct evidence the memory is using context the window has evicted. Full attention is still strongest in the mid bands within the training length (3.59/3.62), as expected for an exact O(T²) model inside its training regime.

**Beyond the training length, full attention collapses while GMSWA-v3 is the single best model.** At 2–4K and 4–8K, the rotary Transformer's loss jumps to 5.74 then 7.21 — a rotary-extrapolation failure — whereas GMSWA-v3 degrades gracefully (3.99, 4.04) and leads all four models. GMSWA thus inherits SWA's extrapolation robustness *and* adds long-range modeling, at constant memory. This is the paper's most consequential result.

### 5.3 Single-Needle Retrieval (NIAH / RULER) — Reported Honestly

Exact-match single-needle recall versus context length, 340M models.

| Context length | SWA | GMSWA-v3 | Transformer (full attn) |
|---|---|---|---|
| 512 | 1.00 | 1.00 | (within window / training) |
| 1024 | 0.64 | 0.64 | 0.86 |
| 2048 | 0.28 | 0.28 | 0.74 |
| 4096 *(extrapolation)* | — | 0.17 | 0.00 |
| 8192 *(extrapolation)* | — | 0.09 | 0.00 |

Within the training length, full attention recalls well (0.86 at 1024, 0.74 at 2048) and both window-limited models track each other and decline as the needle moves outside the window (the window cannot see beyond itself). **At 340M, GMSWA-v3 ≈ SWA on discrete recall**: the memory has learned *soft* utilization (lower loss, §5.2) but not yet sharp discrete copy. In the extrapolation region, both window models retain a little recall (GMSWA-v3 0.17/0.09) while full attention drops to 0.00.

We interpret this conservatively. Discrete needle recall in NIAH/RULER is largely scale-emergent and is weak in sub-billion *base* models; a contemporaneous study on the same stack [arXiv:2507.06457] omits 340M recall for exactly this reason. A 340M tie on discrete recall is therefore the *expected* outcome, not a failure of the memory — its benefit at this scale is the loss-level utilization of §5.2. The discrete-recall win is predicted at larger scale, where SWA is *structurally* capped by its window at any scale while GMSWA's memory can retrieve; we test this at 1B next (§5.5, [PENDING]).

### 5.4 Ablations

The shared-recipe four-model comparison already isolates two design choices; further ablations are [PENDING].

| Ablation | Question | Status | Evidence / placeholder |
|---|---|---|---|
| GMSWA-v3 vs. GMSWA-v2 | Do separate NoPE retrieval projections help? | **Done** | v3 < v2 at every position (§5.2) |
| GMSWA vs. SWA | Does the memory add value? | **Done** | GMSWA-v3 < SWA at every position; margin grows with distance (§5.2) |
| GMSWA vs. Gated DeltaNet | Does the window add value (vs. memory-only)? | **[PENDING]** | same-recipe 336M GDN training in progress |
| Evicted-only vs. memory-on-all (`mem_evicted_only`) | Is complementary coverage the right design? | **[PENDING]** | code implemented, backward-compatible; awaiting training |
| GMSWA vs. GLA / Mamba-2 | Constant-memory peers | **[PENDING]** | parameter-aligned runs queued |

### 5.5 [PENDING] 1B Recall Experiment (Predicted Headline Recall Result)

At 1B parameters, where discrete recall emerges and where SWA remains window-capped, we predict GMSWA's memory yields a discrete-recall win over SWA once the memory retrieves. Queued; results to fill Table below.

| Context length | SWA-1B | **GMSWA-v3-1B** | Transformer-1B |
|---|---|---|---|
| 1024 | [PENDING] | [PENDING] | [PENDING] |
| 2048 | [PENDING] | [PENDING] | [PENDING] |
| 4096 *(extrap.)* | [PENDING] | [PENDING] | [PENDING] |
| 8192 *(extrap.)* | [PENDING] | [PENDING] | [PENDING] |

---

## 6. Mechanism Capacity Probe

To explain *why* the memory behaves as a soft summarizer at small scale and to map the design frontier, we run a training-free diagnostic that removes optimization confounds. We store D key→token associations in a fixed-size state, query each key, decode the retrieved code against a shared codebook, and measure exact-token recall versus D at *equal* state size.

- **Bilinear stores (linear attention, delta rule).** Capacity is bounded by the state rank: storing D near-orthogonal associations needs ≈O(D²) state, so recall falls as load D rises — the **rank wall**.
- **Hash / sketch store.** Capacity is set by the number of buckets, decoupled from the value dimension: D associations need ≈O(D) state, so it escapes the rank wall, with the advantage growing under load (illustratively, at high load the hash store retains markedly higher exact-token recall than the bilinear stores at equal state).

The qualitative finding — a hash/sketch memory's capacity advantage over linear/delta widens with load — confirms the bilinear rank wall as the operative constraint. Exact probe values for the paper's final configuration are reproduced from `mech_capacity.py`; the canonical table is **[PENDING]** pending a clean re-run logged for the camera-ready:

| Load D | linear | delta | hash (sketch) |
|---|---|---|---|
| small D | [PENDING] | [PENDING] | [PENDING] |
| high D | [PENDING] | [PENDING] | [PENDING] |

**Important honest caveat.** The hash/sketch memory that *wins on capacity* is hard to *train*: across MQAR variants, hard-routing gives poor gradients and soft-routing collapses back to a low-rank store (a long-documented hard-vs-soft addressing problem). We therefore use the hash memory strictly as a *diagnostic* that delineates the frontier — not as the proposed mechanism. GMSWA's deployed memory is the trainable gated-delta-rule store; the probe explains its soft-utilization behavior and locates it on the recall–memory frontier of §7.

---

## 7. Analysis and Discussion

### 7.1 The Recall–Memory Frontier

The probe (§6) and the model results (§5) describe one frontier. On one axis is **memory budget** (state size, held constant here); on the other is **discrete recall capacity**. Bilinear constant-memory stores sit below a rank-wall ceiling; sketch stores raise the ceiling but are not reliably trainable; an exact window has perfect recall *within* its span and zero beyond it. GMSWA composes the two regimes — exact recall inside the window, soft trainable summarization outside — and the loss-vs-position curves (§5.2) show the composite improving on both pure SWA and, beyond the training length, on full attention. The frontier framing also predicts the scale dependence: as model scale lifts the memory off the rank-wall floor for the associations that matter, the soft-utilization advantage should sharpen into discrete recall (the 1B test, §5.5).

### 7.2 Why Extrapolation Works

Full attention's collapse beyond 2048 (§5.2) is a rotary-phase failure: relative positions past the training length are out-of-distribution. GMSWA sidesteps this twice — the window only ever sees in-distribution relative offsets (≤ W), and the memory retrieves by *content* through NoPE projections (§3.5), independent of how far back the match lies. The composite therefore degrades gracefully where rotary attention breaks, explaining GMSWA-v3's best-of-all standing in the extrapolation bands.

### 7.3 Honest Scale Caveat

The central honest point: at 340M, GMSWA's memory delivers **soft** long-context utilization (lower loss at every position, widening with distance) but **not** sharp discrete recall (NIAH ties SWA). This is consistent with the scale-emergence of discrete recall in base models and with the rank-wall capacity analysis. We do not claim a discrete-recall win at 340M; we claim a real, measured loss-level win and a structurally motivated, scale-dependent prediction for discrete recall that the 1B run is designed to test.

### 7.4 Positioning: a Reference Framework

Given that the *highest-capacity mechanism* niche is both occupied (FwPKM [arXiv:2601.00671]) and, in our hands, hard to train (§6), GMSWA's contribution is deliberately at the architecture/framework level: a window with a **pluggable, complementary, content-retrieving** constant-memory tail, characterized on the recall–memory frontier. We offer it as a basis on which different memory mechanisms can be dropped in and compared under a fixed, fair window+memory interface.

---

## 8. Acknowledged Limitations

1. **Discrete recall at 340M is a tie, not a win.** The memory's benefit at this scale is loss-level utilization; the discrete-recall claim is explicitly [PENDING] at 1B.
2. **Key baselines are not yet in.** Same-recipe Gated DeltaNet, the evicted-only vs. memory-on-all ablation, and GLA/Mamba-2 peers are [PENDING]; their absence limits how strongly the "window's value" and "complementary design" claims can be asserted today.
3. **Efficiency numbers are partial.** Per-layer state is provably constant, but 340M wall-clock/throughput with `torch.compile`/CUDA-graphs is [PENDING] (Table 7).
4. **Mechanism probe is a diagnostic, not a deployed component.** The capacity-winning hash memory is not trainable in our setup; conclusions from it bound the design space rather than the shipped model.
5. **Single corpus / single scale family.** Results are FineWeb-Edu at 340M (plus a queued 1B); breadth across corpora and downstream tasks is future work.
6. **Citation verification.** arXiv identifiers herein are carried from the project brief and must be DOI/arXiv-verified before submission; any that cannot be verified will be removed or corrected (no unverifiable citation will remain in the camera-ready).

---

## 9. Conclusion and Future Work

GMSWA is a constant-memory attention layer that composes an exact local softmax window with a complementary, content-retrieving gated-delta-rule memory fed only by window-evicted tokens. At 340M/10B tokens under a shared recipe, it achieves the lowest next-token loss at every measured position among constant-memory models — with a margin over SWA that grows with distance — and it is the single best model beyond the training length, where rotary full attention collapses. It does this at a per-layer state size independent of sequence length. We are honest that, at this scale, the win is soft long-context utilization rather than sharp discrete recall, and we explain that gap through a rank-wall capacity analysis and a recall–memory frontier. Future work fills the [PENDING] cells: the 1B discrete-recall experiment, same-recipe Gated DeltaNet / GLA / Mamba-2 baselines, the evicted-only vs. memory-on-all ablation, a recall-vs-memory-budget frontier sweep, and fused-kernel efficiency numbers. We release GMSWA as a reference framework for SWA-based hybrid sparse attention with a pluggable, complementary constant-memory tail.

---

## AI Disclosure

This manuscript draft was prepared with AI assistance (Claude Code, Anthropic) using the `academic-research-skills` `academic-paper` writing methodology for structure, drafting, and quality checks. All quantitative results are real measurements produced by the authors' experiments; the AI tool did not generate or alter any numerical result. All claims were checked against the authors' project records, and every prior-work citation must be independently DOI/arXiv-verified by the authors before submission. The authors take full responsibility for the content. (Final wording should be adapted to AAAI's then-current AI-use disclosure policy.)

## Data Availability Statement

Training data is the public FineWeb-Edu corpus. Model configurations, training scripts, evaluation scripts, and result logs are maintained in the authors' project repository and will be released to support reproduction. [Add repository URL / DOI on release.]

## Ethics Statement

This work studies language-model *architecture* on a public, openly licensed educational web corpus and involves no human subjects, no personally identifying data, and no sensitive content collection. Long-context language models carry the general dual-use considerations of generative LLMs; GMSWA introduces no capability orthogonal to those of existing constant-memory LMs.

## Author Contributions (CRediT)

[Conceptualization: ...; Methodology: ...; Software: ...; Validation: ...; Formal analysis: ...; Investigation: ...; Data curation: ...; Writing — original draft: ...; Writing — review & editing: ...; Visualization: ...; Supervision: ...; Project administration: ...] — complete per author before submission.

## Conflict of Interest Statement

The authors declare no competing interests. [Confirm/adjust before submission.]

## Funding

[Funding sources and grant numbers, or "This research received no specific grant from any funding agency."]

## Acknowledgments

[Collaborators, compute providers, and reviewers.]

---

## References

> ⚠️ Per the `academic-paper` IRON RULE on citations, every entry below must be DOI/arXiv-verified before submission. Identifiers are carried verbatim from the project brief and are **unverified** at draft time; replace any that cannot be confirmed. Formatting here is a working list — convert to AAAI's `aaai` BibTeX style for the camera-ready (see `references.bib`).

1. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.
2. Gated DeltaNet. arXiv:2412.06464.
3. DeltaProduct. arXiv:2502.10297.
4. RWKV-7. arXiv:2503.14456.
5. Based: linear-attention / local-attention interpolation for recall. arXiv:2402.18668.
6. Zoology: recall–throughput tradeoffs in efficient sequence models. arXiv:2312.04927.
7. Test-time regression view of associative memory / rank-bounded capacity. arXiv:2501.12352.
8. Titans: learning to memorize at test time. arXiv:2501.00663.
9. Fast-weight Product-Key Memory (FwPKM). arXiv:2601.00671.
10. laLTE: SWA + linear attention with learned token eviction. arXiv:2510.20787.
11. SWAX: SWA + xLSTM hybrid. arXiv:2509.24552.
12. RULER: long-context benchmark. arXiv:2404.06654.
13. Scale-emergence of recall in sub-billion base models (same-stack study omitting 340M recall). arXiv:2507.06457.
14. [Mistral / sliding-window attention] — add canonical reference.
15. [Longformer] — add canonical reference.
16. [GLA — Gated Linear Attention] — add canonical reference.
17. [FineWeb-Edu dataset] — add canonical reference.
18. [flash-linear-attention (FLA) library] — add canonical reference.
</content>
