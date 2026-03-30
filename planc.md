### 📥 Copy and Paste the following to Codex

**System Role:**
You are an expert AI Research Engineer specializing in Efficient LLM Architectures and Edge AI deployment. You are proficient in PyTorch, Triton, and optimizing Attention mechanisms (FlashAttention, Ring Attention).

**Objective:**
Implement a novel attention mechanism called **"Gated Memory-Augmented Sliding Window Attention" (GM-SWA)**.
**Goal:** mitigate the "context collapse" issue of standard SWA (Sliding Window Attention) by preserving historical information that slides out of the window into a compressed "Global Memory Token."

**Core Philosophy:**
Combine the high-resolution local focus of **SWA** with the infinite context recurrence of **Linear Attention/RNNs** (similar to Mamba/Gated Delta Net), while maintaining  inference memory overhead for Edge NPU deployment.

---

### **Technical Specification**

#### **1. Mathematical Formulation**

Let  be the window size. Let  be the current input.
Standard SWA only attends to keys .
**GM-SWA** introduces a recurrent memory state  (or a matrix state depending on implementation, let's start with a vector state per head for efficiency).

**A. The Gating Mechanism (The "Fancy" Part):**
When a token key/value pair  slides *out* of the window (at step ), we do not discard it. We fuse it into the global memory .

*Note: In this simplified version, we treat the Memory as a "Virtual Key" that summarizes the past.*

**B. The Attention Calculation:**
For the current query :

1. **Local Context:**  (The standard sliding window).
2. **Global Context:** The state  is treated as a special "Sink Token" at index 0.

#### **2. Implementation Requirements (PyTorch)**

Please implement a `GatedMemSWA` class inheriting from `nn.Module`.

**Key Components:**

1. **`__init__`**:
* Args: `dim`, `num_heads`, `window_size`.
* Layers: Standard `q_proj`, `k_proj`, `v_proj`, `o_proj`.
* **New Layer**: `gate_net` (a lightweight Linear layer: `dim` -> `num_heads`).
* **New Layer**: `mem_proj` (Linear layer to project exiting tokens into state space).


2. **`forward` (Training Mode)**:
* Input: `x` (Batch, SeqLen, Dim).
* *Simplification for Training:* For parallel training, you can approximate the recurrence or implement a "Chunkwise" approach. However, for this MVP, assume we process the sequence. If fully parallel implementation is too complex, implement a naive loop or a chunk-based scan for the memory update.
* **Crucial:** Apply RoPE (Rotary Embedding) only to the `window` tokens. The `Memory Token` should effectively be at "position 0" (fixed) to act as a stable anchor.


3. **`inference_step` (Inference Mode - The Priority)**:
* Input: `x_t` (Batch, 1, Dim), `kv_cache` (RingBuffer), `memory_state` (Batch, Heads, HeadDim).
* **Logic:**
* Calculate  for current token.
* Check if KV Cache is full.
* **If full (Sliding):** Retrieve the oldest item  that is about to be overwritten.
* **Update State:** Update `memory_state` using `gate_net(x_t)` and .
* **Concat & Attend:** Prepend `memory_state` to the current `kv_cache` for Attention.
* Update `kv_cache` with new token.





#### **3. Constraints & Optimization**

* **No "Forced" RoPE on Memory:** Do not rotate the Memory State vector based on the current timestep . It must remain rotation-invariant (position 0) so the model can always attend to it easily.
* **Efficiency:** Use `torch.nn.functional.scaled_dot_product_attention` (SDPA) where possible.
* **Edge Focus:** Avoid complex dynamic control flows. Keep matrix multiplications dense.

---

### **Action Plan for Codex**

1. Define the `GatedMemSWA` class structure.
2. Implement the `inference_step` first (as this is the most critical logic for the user's Edge/SWA context).
3. Implement the `forward` pass (Chunkwise or sequential) for compatibility.
4. Write a simple "sanity check" script:
* Run a sequence of length `2 * window_size`.
* Compare the output against a standard SWA (which should degrade) vs GM-SWA (which should maintain coherence).



**Start coding now.**