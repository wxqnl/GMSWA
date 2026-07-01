# GMSWA paper revision plan · submission-oriented

## Goal

Revise the GMSWA paper from the earlier “constant-cache long-context recall” framing into the sharper and more defensible claim:

> Sliding-window attention retains local retrieval but is structurally weak at long-range state tracking; recurrent memory can supply that missing step-wise state, but naive hybrid gates collapse to local-attention shortcuts. GMSWA diagnoses this gate-collapse failure and fixes it with a memory-biased gate initialization.

## What changes from the old draft

1. **Title and thesis**
   - Old: constant-cache long-context modeling + smooth-vs-sharp recall limitation.
   - New: restoring state tracking to sliding-window attention; gate-collapse diagnosis and fix.

2. **Conceptual axis**
   - Old: real recall vs synthetic retrieval.
   - New: retrieval vs state-tracking.
   - NIAH remains in the paper as a retrieval-axis non-regression / tradeoff check, not as the main target.

3. **Method section**
   - Add `mem_beta_range=2.0` as the negative-eigenvalue switch.
   - Add multi-scale memory as the practical memory variant.
   - Add `mix_gate_logit_bias=-4.0` as the gate-collapse fix.
   - Demote SWA-drop curriculum to optional/tuned; it is not part of the core method because bias-only solves parity and S_3 and curriculum hurts S_3.

4. **Experiment section**
   - Keep earlier 340M LM results as retrieval-axis / deployment checks.
   - Add leak-free synthetic state-tracking harness as the central evidence.
   - Add v11gb scale-check results honestly: stable training but mild NIAH/recall regression.

5. **Claims to make**
   - Strong claim: gated-delta memory with β∈(0,2) can track state in our harness; β∈(0,1) cannot.
   - Strong claim: naive hybrid failure is gate-mediated, not memory incapacity; force α=0 recovers perfect parity.
   - Strong claim: gate-bias initialization fixes synthetic gate collapse.
   - Moderate claim: the real-LM gate-bias variant trains stably.
   - Honest caveat: real-LM retrieval-axis results regress mildly; tuning is needed to balance retrieval and state-tracking.

6. **Claims not to make**
   - Do not claim NIAH/RULER improvement.
   - Do not claim state-of-the-art long-context performance.
   - Do not claim the synthetic state-tracking gain has already been demonstrated in natural language tasks.
   - Do not hide v11gb regression.

## New paper structure

1. Introduction
2. Background: Retrieval vs state tracking
3. GMSWA and the gate-collapse problem
4. Leak-free synthetic state-tracking benchmark
5. Experiments
   - Retrieval-axis LM scale checks
   - State-tracking causal chain
   - Generalization to S_3 and latch
6. Discussion
7. Limitations
8. Reproducibility / statements

## Files created

- Markdown submission draft: `GMSWA_STATE_TRACKING_SUBMISSION_2026-06-25.md`
- Updated bib entries: `mozer2026topological`, `grazzi2024unlocking`

## Remaining before actual submission

- Convert the Markdown draft into the target venue's LaTeX style.
- Add multi-seed error bars for synthetic tasks.
- Add gate-routing visualization if time permits.
- Add one more natural-language-like state-tracking task if time permits.
- Run citation-check once venue and final reference list are frozen.
