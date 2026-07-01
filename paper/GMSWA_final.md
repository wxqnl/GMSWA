# Gated-Memory Sliding-Window Attention: Constant-Cache Long-Context Modeling with Recurrent-Level Real Recall

## Abstract

Long-context language modeling is bounded by a memory wall: softmax attention's key–value (KV) cache grows linearly with context, reaching tens of gigabytes at 128K tokens even for sub-billion-parameter models. Sliding-window attention (SWA) bounds the cache but discards information beyond the window; recurrent sequence models (linear attention, SSMs, gated DeltaNet) keep a constant-size state with strong recall, yet trail softmax attention on core language quality. We ask whether a hybrid—a sliding window backed by a constant-size recurrent memory—can simultaneously retain softmax-level base quality, recurrent-level recall, and a constant cache. We introduce **GMSWA** (Gated-Memory Sliding-Window Attention): exact local SWA fused, through a learned per-head gate, with a gated-delta-rule recurrent matrix memory equipped with an induction short-convolution. Under a controlled 340M-parameter comparison with matched training, GMSWA (i) **preserves softmax base quality**, scoring on par with the softmax baselines and above a parameter-matched gated-DeltaNet (GDN) on perplexity-vs-position and a 10-task zero-shot suite; (ii) is **comparable to GDN on real recall-intensive tasks**, exceeding it on SQuAD and FDA while trailing on SWDE; and (iii) attains **constant memory and a constant-time recurrent state**, with a KV cache ~790× smaller than full attention at 128K tokens (and ~11× smaller total inference memory). We further report a careful **negative result**: on *synthetic single-needle* retrieval (NIAH), GMSWA tracks SWA and trails the pure recurrent model, and memory-directed controls fail to close the gap. A memory-only-from-scratch control resolves the main mechanism: removing the SWA branch during training collapses NIAH (0.011 at 2K) and real recall (mean 0.050), so the gap is not explained by window credit assignment alone. It points instead to limited sharp addressing in this gated-delta memory configuration, which we distill into a "smooth-vs-sharp recall" design caution for window–memory hybrids. Our central contribution is this controlled characterization and negative result rather than the hybrid architecture per se.

---

## 1. Introduction

### 1.1 Motivation

Transformer inference over long contexts is dominated by the KV cache. Because softmax attention attends to all previous tokens, its cache grows linearly with sequence length $L$: in our setup a 340M-parameter model needs ~12.9 GB of cache at $L{=}128\text{K}$, and decode latency per token grows with $L$ as each step attends over the entire history. This memory wall, not parameter count, is what makes long-context deployment expensive.

Two families bound this cost. **Sliding-window attention (SWA)** keeps only the last $W$ keys/values, giving a constant cache and exact local attention, but it is structurally blind beyond the window—no amount of scale lets SWA recall a fact $W{+}1$ tokens back. **Recurrent sequence models** (linear attention, state-space models, and gated DeltaNet) compress all history into a fixed-size state, achieving constant memory and strong recall; but across our experiments and the broader literature, they trail softmax attention on core language-modeling quality.

This motivates a natural question:

> *Can a sliding window, backed by a constant-size recurrent memory, keep softmax-level base quality and a constant cache while recovering recurrent-level recall?*

### 1.2 Approach

We study this question with **GMSWA**, which runs, in each layer, (a) exact SWA over a fixed window and (b) a gated-delta-rule recurrent matrix memory over the sequence, and mixes the two through a learned per-head gate. The memory uses an induction short-convolution and dedicated content-retrieval projections so it can address content the window has dropped. Both branches have length-independent state, so the layer—and the model—has a **constant KV cache**.

### 1.3 Findings and contributions

A controlled, parameter-matched comparison at 340M (identical data, tokenizer, steps, and context) yields a nuanced but consistent picture, which we report honestly.

1. **A controlled characterization (primary).** Under a parameter-matched, same-data, same-recipe 340M comparison, we separate three notions of quality that are usually conflated—base modeling, *real* recall, and *synthetic* recall—and show where a window–memory hybrid lands on each: GMSWA preserves softmax base quality (on par with the softmax baselines; above the recurrent GDN, 0.499 vs 0.486), is comparable to GDN on real recall (wins SQuAD/FDA, trails SWDE), and has constant memory with a KV cache ~790× smaller than full attention at 128K (§5).

2. **A negative result and analysis (the most distinctive contribution).** On *synthetic single-needle* NIAH, GMSWA tracks SWA and trails GDN even within the training context. Interventions intended to force the memory to learn sharp retrieval (GDN-style output normalization, pathway dropout, a memory-first curriculum, and memory-only training from scratch) all hurt or collapse. The evidence points to limited sharp addressing in this memory configuration, which we articulate as a *smooth-vs-sharp recall* design caution for window–memory hybrids (§6).

3. **Architecture.** GMSWA itself: SWA fused with a gated-delta recurrent memory (induction short-conv, NoPE retrieval projections, learned per-head gate), giving a constant cache (§3). We position this within the existing local-attention-plus-recurrence family (§2) rather than as a wholly new design.

We deliberately do **not** claim state-of-the-art recall. The value is a clean, controlled study of exactly where and why a window–memory hybrid's recall holds (real, semantic) and stops (synthetic, single-needle).

---

## 2. Related Work

**Sparse and sliding-window attention.** Local/windowed attention and block-sparse variants (Longformer, BigBird, Mistral-style SWA) bound the cache and compute. They lose exact access beyond the window; GMSWA augments SWA with a recurrent memory to recover long-range content while keeping the window's exactness locally.

**Linear attention, SSMs, and gated DeltaNet.** Linear attention, state-space models (Mamba/Mamba-2), and the delta-rule family (DeltaNet, Gated DeltaNet) maintain a constant-size recurrent state. Gated DeltaNet (GDN) is our recurrent baseline and the source of the memory kernel we use. These models recall well but, in our controlled study, trail softmax attention on base quality—the gap GMSWA's window is meant to close.

**Window-plus-recurrence hybrids.** Combining local attention with a global recurrent/linear branch is an active line: Griffin/Hawk (local attention + gated linear recurrence), Samba (Mamba + SWA), Jamba and Zamba (interleaved attention/SSM blocks), RecurrentGemma, YOCO (decoder–decoder with a single global cache), and SWA + linear-attention hybrids such as SWAX; Based and Hymba similarly mix short convolutions/linear attention with local attention. GMSWA is architecturally a member of this family—exact SWA fused per-head with a gated-delta recurrent memory—and we do not claim the block is fundamentally new. Our contribution relative to this body of work is twofold and methodological: (i) a *controlled, parameter-matched* characterization that deliberately separates base modeling, real recall, and synthetic recall—dimensions these papers typically report only in aggregate—and (ii) a *systematic negative result* (five failed interventions) clarifying which kind of recall such hybrids can and cannot acquire under standard LM training, which to our knowledge has not been isolated before.

**Recall benchmarks.** MQAR and Zoology framed the recall–memory trade-off; RULER/NIAH provides synthetic single-needle stress tests; SWDE/FDA/SQuAD probe real extractive recall. We report both real and synthetic recall and show they diverge sharply for window–memory hybrids.

---

## 3. Method

### 3.1 Overview

For input $x\in\mathbb{R}^{B\times T\times d}$, each GMSWA layer computes a local output $o_{\text{loc}}$ (exact SWA) and a memory output $o_{\text{mem}}$ (recurrent), and mixes them per head:
$$o = \alpha \odot o_{\text{loc}} + (1-\alpha)\odot o_{\text{mem}}, \qquad \alpha=\sigma(W_\alpha x)\in(0,1)^{H}.$$

### 3.2 Local branch — exact sliding-window attention

Standard multi-head attention with RoPE, restricted to a causal window of size $W{=}512$. Keys/values are held in a windowed ring buffer, so the local cache is $O(W)$, independent of $L$.

### 3.3 Memory branch — gated-delta recurrent matrix memory

The memory is a gated-delta-rule fast-weight matrix $S\in\mathbb{R}^{H\times d_h\times d_h}$ updated per token with an input-dependent decay $g$ and write gate $\beta$, read by a content query:
$$S_t = \big(\mathrm{diag}(g_t)\big)\,S_{t-1} + \beta_t\, (v_t - S_{t-1} k_t)\,k_t^\top,\qquad o_{\text{mem},t}=S_t\, q_t.$$
Two design choices make this a *retrieval* memory rather than a smoother: (i) **dedicated NoPE retrieval projections** for $q,k$ (decoupled from the positional SWA projections), with L2-normalized keys; and (ii) a **causal depthwise short-convolution** (kernel 4, SiLU) applied to the memory's $q,k,v$—the induction primitive that delta-rule recall relies on. The recurrent state $S$ and the conv state are fixed-size, independent of $L$.

### 3.4 Coverage and the role of the gate

The memory runs over the full sequence; the learned per-head gate decides, per token and head, how much to trust the exact window versus the compressed memory. Empirically the gate relies on the memory heavily (mean SWA weight $\alpha\approx0.21$), i.e., the memory is not idle—a fact that matters for the analysis in §6. A complementary *evicted-only* variant (the memory ingests only tokens that have left the window) is a clean ablation but slightly weaker on real recall.

### 3.5 Constant-cache analysis

The per-layer state is the windowed K/V ring ($O(W)$), the recurrent matrix $S$ ($O(H d_h^2)$), and the conv state ($O(\text{kernel})$)—all independent of $L$. Hence the whole-model cache is **constant in context length**. Concretely (bf16, 24 layers): GMSWA's cache is **16.3 MB at any $L$**, versus full attention's $50\,\text{MB}\to12.9\,\text{GB}$ from $L{=}512\to128\text{K}$ (Table 5).

---

## 4. Experimental Setup

**Models.** All baselines and GMSWA are 340M-parameter (≈369M actual), 24 layers, $d{=}1024$, trained for 10K steps on FineWeb-Edu-100BT with the gla tokenizer (32K vocab), context length 2048, identical optimizer/schedule. Baselines: a full-attention **Transformer** (RoPE $\theta{=}10^6$), **SWA** ($W{=}512$), and a parameter-matched **gated DeltaNet (GDN)** (4 heads × 128, short-conv). This isolates architecture, not scale or data.

**Evaluation.** (i) *Base quality*: token NLL vs position on held-out long documents; a 10-task zero-shot suite (LAMBADA, PIQA, HellaSwag, WinoGrande, ARC-e/c, BoolQ, COPA, OpenBookQA, SciQ). (ii) *Real recall*: SWDE, FDA, SQuAD (contains-accuracy). (iii) *Synthetic recall*: RULER NIAH single-needle at $L\in\{512,...,8192\}$. (iv) *Efficiency*: prefill/decode latency and memory vs $L$ up to 128K (single GPU, bf16). We report token-level NLL rather than word-perplexity, as the latter is unreliable under our tokenizer.

---

## 5. Results

### 5.1 Base quality: GMSWA > recurrent, and length-stable

**Loss vs position (Table 3).** Across buckets to 8K, GMSWA has the lowest NLL among length-stable models (e.g., 3.78 vs SWA 3.87, GDN 3.89 in the 1–2K bucket). The full Transformer is best *within* its training length but **degrades sharply beyond it** (NLL 3.62→7.21 from the 1–2K to 4–8K bucket). We note this reflects RoPE length-extrapolation in our setup (trained at 2K, $\theta{=}10^6$, no test-time extrapolation method) rather than an inherent property of dense attention; it is included to show that, without such methods, dense attention does not natively extend past its training context, whereas GMSWA and the recurrent models remain stable. GMSWA combines low in-range perplexity with this length stability.

**Zero-shot suite (Table 4).** All softmax-based models cluster tightly at ≈0.50 average accuracy and sit **above the recurrent GDN (0.486)**: GMSWA 0.499, SWA 0.498, Transformer 0.500. The within-softmax differences (e.g., GMSWA-v5conv 0.499 vs the earlier GMSWA-v2 0.502) are within single-seed noise and we do not read into them; we ship v5conv as the final configuration for its stronger real-recall behavior (§6), not its zero-shot score. The takeaway is directional and consistent: adding the memory branch does not cost base quality, and the softmax family edges the recurrent baseline. We note these gaps are small and single-seed (see Limitations).

### 5.2 Real recall: GMSWA matches/exceeds the recurrent model

On real recall-intensive tasks (Table 2), **GMSWA is comparable to GDN**, with a 2–1 split: it exceeds GDN on SQuAD (0.284 vs 0.274) and FDA (0.094 vs 0.026) but trails on SWDE (0.088 vs 0.110). All three are far above SWA (0.072/0.040/0.072), so the memory genuinely recovers semantic, real-document recall the window cannot. We do not claim a clean win: the absolute accuracies are low and the per-task gaps are single-seed, so the honest reading is *recall parity with the recurrent baseline*, not superiority. This contrasts sharply with the synthetic-NIAH result (§6), and the contrast is the point: the memory recovers semantic recall but not sharp single-needle retrieval.

### 5.3 Efficiency: constant cache and decode

**Cache and memory (the primary efficiency claim).** GMSWA's KV cache is analytically constant at **16.3 MB** for any $L$, while full attention grows to **12.9 GB at 128K—a ~790× KV-cache reduction** (this is a cache-only ratio; end-to-end inference memory, which includes the model and activations, is ~11× smaller: 1.25 GB vs 14.1 GB at 128K). Figure 1 (right) confirms this: measured inference memory is flat for GMSWA, SWA, and GDN but grows linearly for the Transformer. This panel is clean and is our headline efficiency result.

**Decode latency (Figure 1, left).** GMSWA's per-token decode is **constant in $L$** (~34.5 ms/token from 1K to 128K, modulo a measurement-harness bump near 32K), as is GDN's (~20 ms), whereas the Transformer grows 8.4× (14.5→122 ms). Two honesty notes. (i) GMSWA's decode is ~2× SWA's at short $L$ because it runs two branches per token; the win is constancy, not raw speed. (ii) The **standalone SWA baseline's measured decode grows with $L$** in our harness (15.9→116 ms), tracking the Transformer rather than staying flat. This is an implementation artifact of that baseline's KV cache, which materializes the full history at decode instead of exploiting the window; its *cache size* is nonetheless constant (Figure 1, right; Table 5), and GMSWA's hand-written windowed ring buffer keeps GMSWA's own decode flat. We report SWA's curve as-is rather than hide it; the relevant comparison for our claim is GMSWA vs. full attention, where the constant-vs-linear gap is unambiguous in both panels.

### 5.4 Summary

GMSWA is the only model in our study that is simultaneously (a) competitive on base quality (better than recurrent), (b) competitive on real recall (matches recurrent), and (c) constant in cache and decode latency (unlike full attention). The one axis on which it does not win is synthetic single-needle recall, which we now examine in depth.

---

## 6. The Synthetic-Recall Limitation: A Negative Result and Analysis

### 6.1 The gap

On synthetic single-needle NIAH (Table 1), GMSWA tracks SWA (0.27 vs 0.29 at 2K; 0.16 vs 0.18 at 4K) and trails the pure recurrent GDN (0.57, 0.40). Notably the gap exists even *within* the training context (2K)—i.e., for needles already beyond the 512 window but inside the trained sequence length—so it is not an extrapolation failure: the window cannot see the needle, and the hybrid's memory does not perform the sharp single-token retrieval that would compensate, even though it uses the *same* gated-delta kernel that GDN uses to succeed.

### 6.2 Gate use is real; memory-only probes fail

Gate laziness is not the explanation: the learned gate is **memory-dominant** (mean SWA weight $\alpha\approx0.21$), so the memory is heavily used. Simple rank counting is also insufficient. A single needle appears to require one key→value association, yet forcing the trained mix to memory-only ($\alpha{=}0$) drives NIAH to 0. The window supplies the limited synthetic recall GMSWA shows; the memory branch, as configured, does not perform sharp single-token retrieval.

### 6.3 Memory-directed controls fail (Table 6)

If the memory *could* learn sharp retrieval under the same objective once the window stopped helping, forcing reliance on it should help. We tested memory-directed controls:
- **GDN-style output gated-RMSNorm** on the memory readout (v6): yielded the *best* perplexity of all models but **collapsed recall** (NIAH≈0), because normalization smooths away the sharp retrieval signal.
- **Pathway dropout, aggressive** ($p{=}0.5$, v7): destabilized training (NLL ~7–9).
- **Pathway dropout, gentle** ($p{=}0.15$, v8): degraded everything (NIAH(2048) 0.11 vs the 0.27 baseline; NLL ~5.4).
- **Memory-first curriculum** (anneal drop $1.0\to0$ over 3K steps, v9): a competent LM (NLL ~4.3) but NIAH≈0.
- **Memory-only from scratch** (local branch disabled for the full run): NIAH(2048) 0.011 and real-recall mean 0.050.

Every control that shifts reliance toward this memory hurts or collapses recall.

### 6.4 Explanation: smooth recall and limited addressing

The memory-only-from-scratch run resolves the main ambiguity. We trained the same GMSWA memory branch with the local window architecturally disabled (`disable_local=true`) for the full 10K-step recipe. It did not approach GDN: mean NIAH was 0.071/0.035/0.011/0.010/0.002 at 512/1024/2048/4096/8192, and real-recall mean was 0.050. At 2K, where GDN reaches 0.57 and full GMSWA reaches 0.27, the memory-only model reaches 0.011. Removing the window therefore does not make this memory learn sharp retrieval.

The evidence points to an addressing/capacity limitation of this memory configuration under LM training. The branch can help smooth semantic recall when fused with SWA, but its 16×64 gated-delta state and NoPE/short-conv interface do not support pointer-like single-needle access at 340M. GDN's success shows that recurrence can learn the task; this hybrid memory, as configured, does not.

We distill this as a **design caution**: *window–memory hybrids need a retrieval-explicit objective or sharper, higher-capacity addressing if exact synthetic retrieval is a target; forcing reliance on the existing memory is insufficient.* We scope this to 340M, 10K steps, and this branch design. A per-branch gradient probe and a 1B replication remain useful, but the completed memory-only control removes credit assignment as the central explanation for these results.

---

## 7. Discussion

GMSWA delivers a favorable point in the long-context design space: softmax base quality, recurrent-level *real* recall, and a constant cache, with no scale penalty (340M, controlled). Its limitation is precise and explainable. The memory-directed controls rule out gate laziness and simple reliance failures, and localize the synthetic-recall gap to sharp addressing in the memory branch.

---

## 8. Limitations

(1) **Scale, seeds, and budget.** All results are at 340M with 10K-step training and a **single seed**; the close comparisons (zero-shot 0.499 vs 0.486; the real-recall per-task gaps) are therefore directional, not statistically established—multi-seed runs with confidence intervals are needed and are our top revision priority. Recall is partly scale-emergent, so a 1B-scale study (which we did not complete) could shift the synthetic-recall picture; the negative result is strongest as a statement about this regime. (2) **No external-hybrid head-to-head.** We compare against Transformer/SWA/GDN but did not retrain a literature hybrid (Griffin/Samba/SWAX-style) under our recipe; one such comparison would sharpen the positioning and is planned. (3) **Real-recall tasks are easier** than synthetic single-needle and use low-accuracy regimes; "recall parity" should be read accordingly. (4) **Efficiency measurement.** GMSWA's per-token decode is ~2× SWA's (two branches), albeit constant in $L$; our decode-latency harness shows a regime change near 32K and the standalone-SWA baseline's decode is an implementation artifact (§5.3)—the robust efficiency claims are the analytical KV-cache reduction and the measured-memory panel. (5) **Metric hygiene.** We report token-level NLL because word-perplexity is unreliable under our tokenizer. (6) **The memory-only control is still single-seed and 340M**; per-branch gradient probes and 1B replication remain useful to separate capacity, addressing, and training dynamics.

---

## 9. Conclusion

We presented GMSWA, a sliding-window attention fused with a constant-size gated-delta recurrent memory. In a controlled 340M study it preserves softmax base quality (above a recurrent baseline), is comparable to that baseline on real recall (winning two of three tasks), and keeps a constant cache (~790× smaller KV cache than full attention at 128K). We also showed, through memory-directed interventions and a memory-only-from-scratch control, that synthetic single-needle recall is hard for this hybrid memory under standard LM training at this scale, and traced the failure to limited sharp addressing rather than window credit assignment alone. GMSWA is thus both a practical constant-cache long-context model and a clean case study of where window–memory hybrids' recall comes from—and where it stops. We hope the controlled protocol and the negative result are useful beyond GMSWA itself.
