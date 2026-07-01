# Restoring State Tracking to Sliding-Window Attention with Gated Recurrent Memory

## Abstract

Long-context language modeling is often evaluated as if memory were a single capability. We argue that two capabilities should be separated: **retrieval**, the lookup of static past tokens, and **state tracking**, the iterative update of a latent state over time. Sliding-window attention (SWA) is useful for local retrieval and offers a constant cache, but it is structurally weak for state tracking beyond its window. Recurrent sequence models offer step-wise state, but recurrence-attention hybrids can fail to use the recurrent branch when a local-attention shortcut is available. We study this failure in **GMSWA** (Gated-Memory Sliding-Window Attention), a hybrid layer that fuses exact local SWA with a gated-delta recurrent matrix memory. Motivated by recent work on transformer state-tracking limits and negative eigenvalues in linear RNNs, we add a `beta_range=2.0` transition that permits negative eigenvalues, and we construct leak-free synthetic state-tracking tasks in which the target state is never present in the input stream. The experiments reveal a causal chain: SWA and finite-depth Transformers fail out of range; pure memory with beta in (0,1) fails; pure memory with beta in (0,2) solves parity perfectly; the same memory inside a naive hybrid fails; forcing the hybrid gate to memory recovers perfect performance; and a memory-biased gate initialization fixes the learned hybrid. The pattern generalizes from parity to S_3 group composition. At 340M parameters, the same memory changes train stably and preserve retrieval for the beta/multiscale variant, while the memory-biased gate introduces a mild NIAH/recall regression, exposing a real retrieval--state-tracking calibration tradeoff. Our contribution is therefore not a state-of-the-art long-context model, but a controlled diagnosis and fix for gate collapse in hybrid SWA--recurrence models.

**Keywords:** sliding-window attention; recurrent memory; state tracking; long-context language models; gated DeltaNet; hybrid attention

## 摘要（中文）

长上下文模型的“记忆”能力常被混为一谈。本文区分两类能力：**retrieval**（对历史静态 token 的查找）和 **state tracking**（沿时间递推更新隐状态）。滑动窗口注意力适合局部 retrieval 且缓存恒定，但在窗口外的状态追踪上存在结构性弱点；递归记忆具备逐步状态更新能力，但在与局部注意力混合时，learned gate 容易塌缩到局部注意力捷径，压制 memory 分支。我们以 GMSWA 为研究对象，加入允许负特征值的 `beta_range=2.0` 记忆更新，并构造无泄漏的合成 state-tracking 任务。实验显示：SWA 与有限深度 Transformer 在长程 state-tracking 上失败；beta∈(0,2) 的纯 memory 可完美追踪 parity；同一 memory 放入 naive hybrid 后失败；强制 gate 走 memory 立即恢复；而 memory-biased gate initialization 可修复 learned hybrid。该结论从 parity 泛化到 S_3 群合成。340M 真实 LM scale-check 显示该机制可稳定训练，但 gate-bias 会带来轻微 retrieval/recall 回退，提示 retrieval 与 state-tracking 的校准仍是开放问题。

---

## 1. Introduction

Efficient long-context language models are usually discussed in terms of a single question: how much past information can the model remember under a bounded cache? This framing hides an important distinction. A model may retrieve a static token from the past, or it may maintain a latent state that changes after each observation. These are different computational problems. Retrieval can often be solved by searching over stored tokens. State tracking requires a recurrence of the form

\[
s_t = f(s_{t-1}, x_t),
\]

where the state is not necessarily recoverable from any single previous token.

This distinction matters for sliding-window attention (SWA). SWA is attractive because it bounds the KV cache and preserves exact local attention. It is also widely deployed in practical long-context systems. But a sliding window discards tokens outside the window. Beyond that boundary, the model cannot retrieve old tokens, and it cannot maintain a state unless some recurrent mechanism carries information forward. Recent theoretical perspectives on transformer state tracking make this concern sharper: finite-depth feedforward computation must repeatedly re-encode state through depth, whereas a time-axis recurrence can update state at every step.

Hybrid attention--recurrence models are a natural response. A local attention branch handles local retrieval and language quality; a recurrent branch maintains a compressed state. Yet the hybrid itself introduces a new failure mode: a learned gate may prefer the easy local-attention shortcut during training and suppress the recurrent branch, even when the recurrent branch is the only component capable of solving the long-range state-tracking task.

We study this failure in **GMSWA** (Gated-Memory Sliding-Window Attention), a layer that combines exact SWA with a gated-delta recurrent matrix memory. Earlier versions of GMSWA were motivated by constant-cache long-context modeling and showed a mixed picture: strong base quality and real recall, but no improvement on synthetic needle retrieval. In this revision, we reframe the model around a sharper question:

> Can a sliding-window model recover long-range state tracking through a recurrent memory, and what prevents a naive hybrid from using that memory?

### Contributions

1. **A retrieval/state-tracking reframing of GMSWA.** We separate token retrieval benchmarks such as NIAH from state-tracking tasks such as parity, group composition, and latch. This prevents overclaiming: improving state tracking should not be expected to improve NIAH automatically.
2. **A negative-eigenvalue memory switch.** We add `mem_beta_range`, where beta in (0,2) allows transition eigenvalues in [-1,1]. This implements the mechanism identified by Grazzi et al. as necessary for parity-like state tracking.
3. **A leak-free synthetic state-tracking harness.** The input stream contains operations and a content-free query token; the running state is never shown as input. The model must output the cumulative state at query positions.
4. **A gate-collapse diagnosis.** Pure beta=2 memory solves parity perfectly, but the same memory inside a naive GMSWA hybrid fails. Forcing the mix gate to memory recovers perfect performance, localizing the failure to the gate.
5. **A simple gate-bias fix.** Initializing the mix gate toward memory (`mix_gate_logit_bias=-4`) fixes the learned hybrid on parity and S_3. A SWA-drop curriculum is not part of the core method because bias-only works and the curriculum can hurt S_3.
6. **A real-LM scale check.** At 340M parameters, beta/multiscale changes preserve retrieval-axis performance, while the gate-bias variant trains stably but mildly regresses NIAH/recall. We report this as an honest calibration tradeoff.

![Mechanism overview: retrieval vs state-tracking, gate-collapse diagnosis, and gate-bias fix.](figures/gmswa_mechanism_causal_chain.png)

**Figure 1. Mechanism overview.** Long-context memory separates into retrieval and state tracking. SWA supplies local retrieval; recurrence supplies step-wise state tracking. The key hybrid failure is gate collapse: the learned mix gate can prefer the local SWA shortcut even when recurrent memory is required.

---

## 2. Related Work

**Sliding-window and sparse attention.** Local and block-sparse attention methods such as Longformer, BigBird, and Mistral-style SWA reduce memory and compute by restricting attention patterns. They preserve exact local attention but cannot directly access tokens outside the retained window. GMSWA keeps the local exactness of SWA while adding a recurrent state.

**Linear attention, state-space models, and gated delta rules.** Linear attention, SSMs, Mamba/Mamba-2, DeltaNet, and Gated DeltaNet maintain fixed-size recurrent states. These architectures provide the step-wise recurrence missing from local attention, but they can trail softmax attention on local language quality. GMSWA uses a gated-delta-style memory as the recurrent branch.

**Hybrid attention--recurrence models.** Griffin, Samba, Jamba, Zamba, RecurrentGemma, YOCO, Based, Hymba, and related hybrids combine local/global attention with recurrent or linear components. GMSWA belongs to this family. The novelty claimed here is not merely the block design, but the controlled diagnosis of a hybrid-specific failure: the gate can suppress a capable recurrent branch.

**State tracking.** Mozer et al. argue that feedforward Transformers are topologically ill-suited for maintaining evolving state indefinitely. Grazzi et al. show that linear RNNs require negative eigenvalues to unlock parity-like state tracking. Our experiments connect these ideas to a practical SWA--recurrence hybrid: negative eigenvalues make the memory capable, but the hybrid gate must still route to that memory.

---

## 3. GMSWA

Given hidden states \(x\in\mathbb{R}^{B\times T\times d}\), each GMSWA layer computes a local attention output and a recurrent memory output:

\[
o_t = \alpha_t \odot o_{\mathrm{SWA},t} + (1-\alpha_t)\odot o_{\mathrm{mem},t},
\quad
\alpha_t = \sigma(W_\alpha x_t + b_\alpha).
\]

The local branch is exact causal SWA with window \(W\). The recurrent branch is a gated-delta matrix memory:

\[
S_t = \mathrm{diag}(g_t)S_{t-1} + \beta_t(v_t - S_{t-1}k_t)k_t^\top,
\quad
o_{\mathrm{mem},t}=S_t q_t.
\]

The branch uses separate memory projections, L2-normalized keys, and short convolutions over memory \(q,k,v\), following the gated-delta design.

### 3.1 Negative-eigenvalue switch

In the original memory, beta is produced by a sigmoid and lies in (0,1). For a rank-one delta update, the transition term has eigenvalues constrained to [0,1]. This is insufficient for tasks such as parity that require sign-flipping dynamics. We therefore introduce

\[
\beta_t = R\cdot\sigma(z_t),
\]

where `mem_beta_range=R`. The default \(R=1\) preserves the original behavior; \(R=2\) permits eigenvalues in [-1,1].

### 3.2 Multi-scale memory

We add `mem_num_scales`, with slow and fast decay ranges and a learned scale gate. The slow scale is initialized for long retention, while the fast scale is initialized for rapid local updates. This gives the memory a natural multi-timescale state representation.

### 3.3 Gate-bias initialization

The hybrid output gate can collapse to the local SWA branch during training. We therefore initialize the mix gate toward memory:

\[
b_\alpha = -4.
\]

Since \(\alpha\) is the SWA weight, a negative bias starts the layer closer to the memory branch. The gate remains learned. This is not equivalent to forcing memory at inference; it changes the optimization path so that the recurrent branch is not suppressed early.

---

## 4. Leak-Free State-Tracking Benchmark

We construct synthetic streams of alternating operation tokens and content-free query tokens:

\[
\mathrm{op}_1, Q, \mathrm{op}_2, Q, \ldots, \mathrm{op}_n, Q.
\]

At each query position, the model predicts the cumulative latent state after the preceding operation. The state is never present in the input stream. Loss is applied only at query positions, and logits at a query position predict the state at that same position. This avoids next-token leakage and ensures that a model must either see the relevant operation history or carry state forward.

We report accuracy by operation-distance buckets relative to the SWA window:

\[
\le W/2, \quad W/2..W, \quad W..2W, \quad 2W..4W, \quad >4W.
\]

Tasks:

- **Parity:** one-bit cumulative XOR.
- **S_3 group composition:** cumulative composition over the six elements of the symmetric group S_3.
- **Latch:** a one-bit value is set once and must be carried forward.

All synthetic models use depth 6, width 256, window 64, sequence length 1024, 4000 steps, and a single seed unless otherwise stated.

---

## 5. Experiments

### 5.1 Retrieval-axis LM scale checks

All LM scale checks use a 340M model, 24 layers, hidden size 1024, FineWeb-Edu-100BT, 10K steps, context length 2048, and the same tokenizer and optimizer. The evaluation suite includes RULER/NIAH at lengths 512--8192 and recall tasks SWDE/FDA/SQuAD.

| Model | NIAH@512 | @1024 | @2048 | @4096 | @8192 | Recall avg |
|---|---:|---:|---:|---:|---:|---:|
| v5conv baseline | 0.777 | 0.475 | 0.242 | 0.143 | 0.070 | 0.155 |
| v10ms: multiscale + beta=2 | 0.730 | 0.489 | 0.231 | 0.156 | 0.070 | 0.160 |
| v11gb: v10ms + gate bias | 0.730 | 0.424 | 0.208 | 0.101 | 0.056 | 0.134 |

The v10ms model preserves the retrieval axis: its NIAH/recall results are effectively tied with v5conv. The gate-bias variant v11gb trains stably (final loss 2.5415) but regresses mildly on NIAH and recall. This is important: the state-tracking fix is not a free NIAH improvement. It introduces a calibration tradeoff between retrieval and memory-biased state tracking.

![Retrieval-axis scale check: v5conv, v10ms, and v11gb NIAH curves plus recall averages.](figures/gmswa_retrieval_scalecheck.png)

**Figure 2. Retrieval-axis scale check.** v10ms is essentially tied with the v5conv baseline on NIAH and recall, while v11gb trains stably but mildly regresses retrieval. Gate-bias fixes synthetic state tracking, not NIAH.

### 5.2 Parity: causal mechanism chain

| Variant | ≤W/2 | W/2..W | W..2W | 2W..4W | >4W | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| SWA | 1.00 | 0.77 | 0.50 | 0.50 | 0.50 | window cliff |
| Transformer | 0.98 | 0.61 | 0.50 | 0.50 | 0.50 | finite-depth limit |
| memonly beta=1 | 0.87 | 0.51 | 0.50 | 0.50 | 0.50 | no negative eigenvalues |
| **memonly beta=2** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | memory is capable |
| naive GMSWA beta=2 | 0.94 | 0.52 | 0.50 | 0.50 | 0.50 | gate collapse |
| **force alpha=0** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | gate is bottleneck |
| **gate-bias only** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | learned hybrid fixed |

The causal chain has no counterexample in this setting. The memory branch is sufficient when beta permits negative eigenvalues. The naive hybrid fails because the gate does not route to memory. Forcing memory recovers perfect accuracy, and the learned gate can be fixed by a memory-biased initialization.

### 5.3 S_3 group composition

| Variant | ≤W/2 | W/2..W | W..2W | 2W..4W | >4W | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| SWA | 0.540 | 0.200 | 0.167 | 0.165 | 0.168 | window cliff |
| Transformer | 0.474 | 0.168 | 0.164 | 0.167 | 0.167 | random at long range |
| **memonly beta=2** | **1.000** | **1.000** | **1.000** | **0.998** | **0.972** | recurrent memory tracks state |
| naive GMSWA beta=2 | 0.496 | 0.172 | 0.164 | 0.167 | 0.167 | gate collapse |
| **gate-bias only** | **1.000** | **1.000** | **0.995** | **0.967** | **0.887** | learned hybrid mostly fixed |

The S_3 results show that the parity finding is not just a one-bit artifact. A beta=2 memory nearly solves long-range group composition, while the naive hybrid collapses to chance. Gate-bias recovers most of the long-range performance.

![Synthetic state-tracking results for parity and S_3.](figures/gmswa_synthetic_state_tracking.png)

**Figure 3. Synthetic state-tracking results.** In parity and S_3, SWA and finite-depth Transformers fall to chance at long distances. Pure beta=2 memory remains accurate, but naive GMSWA collapses because the learned gate suppresses memory. Gate-bias restores learned state tracking.

### 5.4 Latch

| Variant | ≤W/2 | W/2..W | W..2W | 2W..4W | >4W | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| SWA | 1.0 | 1.0 | 1.0 | 0.73 | 0.49 | layer relay, then depth exhaustion |
| Transformer | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | easy one-bit carry |
| memory / GMSWA | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | distance-invariant |

Latch is easy compared with parity and S_3. It is still useful because SWA shows the expected depth-relay behavior before collapsing at long distances, while recurrent memory remains distance-invariant.

![Latch behavior and task-validity summary.](figures/gmswa_latch_validity.png)

**Figure 4. Latch and task validity.** Latch is easy for full attention and recurrent memory but still exposes SWA's depth-relay collapse. The validity map records which synthetic tasks were retained or rejected.

---

## 6. Discussion

### 6.1 Gate collapse is a hybrid-specific failure

The main result is not simply that recurrence helps state tracking. Pure recurrence with beta=2 does help, but the same recurrence inside a hybrid can fail. The failure is produced by the interaction between the local branch and the gate. During training, local SWA offers an easier short-range shortcut. The gate can therefore learn to rely on SWA even on tasks whose solution ultimately requires recurrence.

This explains why naive hybridization is not enough. A recurrent branch may be mathematically capable, but optimization can prevent the model from routing to it.

### 6.2 Retrieval and state tracking should not be conflated

NIAH is a retrieval benchmark. It asks whether a model can recover static tokens from a long context. Parity and S_3 require state tracking: the target is not present as a token and must be computed by integrating operations. The v11gb scale-check shows why this distinction matters. Biasing the gate toward memory fixes synthetic state tracking but mildly hurts NIAH and recall. A future production model should therefore tune the gate to preserve retrieval while enabling state tracking, rather than treating one benchmark as a proxy for all memory.

### 6.3 Why the gate-bias result is still useful

Even with the v11gb retrieval regression, the gate-bias result is valuable. It proves that the learned hybrid can be made to use the recurrent state. The remaining problem is calibration, not capability: how much memory bias is needed, whether the bias should be annealed, and whether gate regularization can encourage memory use only when state tracking is needed.

---

## 7. Limitations

1. **Single seed.** All synthetic and 340M scale-check results are single-seed. The causal gaps are large, but final numbers still need error bars.
2. **Synthetic tasks.** Parity and S_3 isolate state tracking, but they are not natural language. A natural-language-like entity-state or task-phase benchmark is needed.
3. **Retrieval regression in v11gb.** The gate-bias scale-check regresses NIAH and recall mildly. We do not present it as a finished deployment recipe.
4. **No gate-activation visualization yet.** The current diagnosis uses causal interventions; visualizing alpha distributions would strengthen the mechanism story.
5. **340M and 10K steps only.** Larger models or longer training may alter the retrieval/state-tracking balance.
6. **External hybrid baselines not retrained.** We compare against our controlled Transformer/SWA/GDN baselines but not against every recent hybrid architecture under the same recipe.

---

## 8. Conclusion

GMSWA shows that restoring recurrence to sliding-window attention is not just an efficiency trick. It exposes a deeper optimization issue in hybrid memory models: a capable recurrent memory can be suppressed by a learned gate that prefers local-attention shortcuts. By separating retrieval from state tracking, adding a negative-eigenvalue memory update, and using leak-free synthetic diagnostics, we identify the failure and show that a simple memory-biased gate initialization fixes it in controlled state-tracking tasks. The real-LM scale check is more cautious: the same bias trains stably but mildly hurts retrieval, making calibration the next step. The resulting paper is therefore a mechanism paper: it diagnoses and fixes gate collapse for state tracking, while clearly delimiting what remains unsolved for retrieval.

---

## Reproducibility and Statements

**Data availability.** Training used FineWeb-Edu-100BT and standard lm-evaluation-harness/RULER-style tasks. Synthetic task generators and evaluation scripts are in `/data/Minko/st_gen_tasks.py` and `/data/Minko/st_train_eval.py`; experiment records are in `/shared/Minko/GMSWA/paper/STATE_TRACKING_EXPERIMENTS.md`.

**Code availability.** The relevant GMSWA implementation is in `flash-linear-attention/fla/layers/gated_mem_swa.py` and `flash-linear-attention/fla/models/gated_mem_swa/`. The v11gb checkpoint and logs are under `/data/Minko/GMSWA/flash-linear-attention/flame/saves/GMSWA-340M-v11gb-10k/` and `/data/Minko/GMSWA/eval_results/suite/GMSWA-v11gb/`.

**Ethics statement.** This work uses public text corpora and synthetic diagnostic tasks. It does not involve human subjects, private user data, or deployed decision-making systems.

**Author contributions.** Anonymous for review. Conceptualization: project authors. Methodology: project authors. Software: project authors. Investigation: project authors. Writing: project authors.

**Conflict of interest.** The authors declare no known conflicts of interest.

**Funding.** Funding information is omitted for anonymous review and should be filled before camera-ready submission.

**AI assistance disclosure.** AI tools were used to assist with drafting, editing, code navigation, and experiment-log summarization. All claims, numerical results, and citations must be verified by the authors before submission.

---

## AAAI submission checklist

- [x] Main AAAI-style LaTeX draft created: `GMSWA_STATE_TRACKING_AAAI_2026_06_25.tex`.
- [x] Figure PDFs/PNGs created under `paper/figures/`.
- [x] Citation keys checked against `references.bib`; no missing keys in the current LaTeX draft.
- [x] Core final eval numbers inserted: v5conv / v10ms / v11gb NIAH and recall.
- [x] Synthetic state-tracking tables inserted: parity, S_3, latch.
- [x] Scope caveat inserted: v11gb fixes synthetic state tracking but mildly regresses retrieval/recall.
- [ ] Compile the LaTeX source in an environment with `pdflatex` and `bibtex` installed.
- [ ] Run final citation/integrity check after compilation.
- [ ] Trim to AAAI page limit after seeing compiled page count.
- [ ] Add multi-seed error bars if time allows.
- [ ] Add real gate-activation/routing visualization if time allows; current figures show the causal mechanism, not measured alpha histograms.
- [ ] Fill author/funding metadata after de-anonymization.

---

## References

- Mozer, M. C., Siddiqui, S. A., & Liu, R. (2026). *The Topological Trouble With Transformers*. arXiv:2604.17121.
- Grazzi, R., Siems, J., Zela, A., Franke, J. K. H., Hutter, F., & Pontil, M. (2024). *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*. arXiv:2411.12537.
- Additional architecture and benchmark references are maintained in `references.bib`.
