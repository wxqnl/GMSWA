# GMSWA paper — deferred experiments (TODO)

The paper text (`GMSWA_AAAI_final.tex`) is complete and compiles. The items below
are **GPU-gated** (the cluster is currently running an unrelated job). Each is
flagged in-line in the `.tex` with `\todo{...}`. Priority order reflects impact
on the reviewer verdict (currently *weak accept, text-conditional*).

---

## A. ★ Memory-branch-from-scratch (CORE — converts the central claim from hypothesis to evidence)

**Why it's the most important.** Our headline analysis (§6.4) claims the
gated-delta memory *could* learn sharp single-needle retrieval but doesn't,
because the SWA window pre-empts the gradient signal (credit assignment). The
five negative interventions are *consistent* with this but do not *prove* it
over the alternative "the memory kernel simply can't address sharply enough."
This single experiment discriminates them and is the evidential backbone of the paper.

**Status: COMPLETED (2026-06-11).**
- `disable_local` flag implemented in `fla/layers/gated_mem_swa.py` (+ config +
  modeling), tested: memory-only prefill/decode cosine 1.0, trains, SWA output bypassed.
- Run: `GMSWA-340M-memonly-10k`, config `disable_local: true`, trained to step 10000 on node41.
- Eval artifacts: `eval_results/suite/GMSWA-memonly/{ruler_512,...,ruler_8192,recall}`.
- Eval harness fixes made before the valid run: `/data` root path, offline HF dataset cache
  (`baber/paul_graham_essays`, `hazyresearch/based-{swde-v2,fda,squad}`), NLTK
  `punkt_tab`, `wonderwords`, and sourcing the venv CUDA shim for Triton `-lcuda` linking.
- Final NIAH mean over 5 tasks: 512=0.071, 1024=0.035, 2048=0.0105, 4096=0.0095, 8192=0.002.
- Final real-recall: FDA=0.010, SQuAD completion=0.086, SWDE=0.054 (mean=0.050).

**Outcome / interpretation.**
Memory-only **also fails NIAH**, and it is worse than the full hybrid at every length
(e.g. @2048: memonly 0.0105 vs full GMSWA about 0.24 vs GDN 0.57). This falsifies
A's credit-assignment hypothesis at 340M: removing the SWA window did not uncover a
GDN-like retrieval memory. The bottleneck is memory capacity/addressing (16×64 state,
NoPE, short-conv/gated-delta configuration), not merely the local window pre-empting
training signal. Revise §6.4 to this capacity/addressing story.

**Companion (cheap):** per-branch gradient-norm during training (does the memory's
retrieval pathway receive vanishing gradient on the needle token?) + a per-layer
memory-only retrieval probe. ~1 training run + logging.

---

## B. ★ Multi-seed + confidence intervals (top stats priority; reviewer's #1 gate)

Every close comparison is single-seed: zero-shot 0.499 vs 0.486; SQuAD 0.284 vs
0.274; FDA 0.094 vs 0.026; SWDE 0.088 vs 0.110. Re-run **GMSWA and GDN (≥3 seeds)**
on the zero-shot suite, the 3 real-recall tasks, and NIAH@2K; report mean ± CI in
Tables `tab:zs`, `tab:recall`, `tab:niah`. If full retrains are too costly, at
least bootstrap CIs on the eval sets (cheap, no retrain). Until this exists, every
"above/wins" must stay worded as "on par/comparable" (already done in text).

---

## C. Verify all citations (cheap, no GPU)

`references.bib` lists real, well-known papers but the arXiv IDs / authors / venues
are from memory. Verify each against the official source (esp. **swax2025**, whose
ID 2509.24552 and authorship are explicitly unverified). Required by the
academic-paper IRON RULE before any submission.

---

## D. One external-hybrid head-to-head (strengthens novelty/positioning)

Retrain **one** literature hybrid (a Griffin-style gated-linear-recurrence+local
block, or a Samba/SWAX-style block) under the identical 340M recipe and add it to
Tables 1–4. Even one external comparable converts "vs our own baselines" into "vs
the field" and directly answers the novelty critique.

---

## E. 1B-scale replication of the negative result (generality)

Recall is partly scale-emergent; the negative result is currently scoped to 340M.
A 1B GMSWA + 1B GDN run (the 1B training was started earlier and stopped on the
diagnosis) would test whether the synthetic-recall gap persists at scale. Expensive
(~13–15h on 8 GPUs each). Lower priority than A/B.

---

## F. Short-conv ablation (cheap, supports a Method claim)

§3.3 states the induction short-conv is "the induction primitive that delta-rule
recall relies on." Isolate its contribution: GMSWA with vs without the memory
short-conv, on NIAH and real recall. (We have the flag `mem_use_short_conv`; v3 is
the no-conv variant, v5conv has it — partially available; one clean run completes it.)

---

### Status snapshot
- Text: **complete**, compiles to a clean 6-page AAAI PDF, 2 review rounds passed
  (verdict: weak accept, text-conditional).
- Blocking a higher verdict: **B** (statistics). **A** is complete and now requires §6.4 text revision.
- Code already in place: `disable_memory`, `mem_use_short_conv`, `mem_use_output_norm`,
  `mem_swa_drop_prob(+anneal)`, `mem_evicted_only` flags in `fla/layers/gated_mem_swa.py`.
  The **A**-specific `disable_local` flag has been trained/evaluated; next priority is **B**.
