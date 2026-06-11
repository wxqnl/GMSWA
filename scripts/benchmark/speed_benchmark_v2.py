"""Polished long-context efficiency benchmark for the paper.
Chunked prefill (so constant-cache models reach 128k without prefill-activation OOM),
kernel warmup, and steady-state cache memory measured AFTER prefill. Produces:
  - decode latency / token  vs  context length
  - steady-state memory (model + KV cache)  vs  context length
  - prefill throughput  vs  context length
Full attention (Transformer) OOMs once its O(L) cache blows the budget — that's the point.
Single GPU, bf16. Saves a 3-panel figure to paper/figures/efficiency.pdf|png + JSON.
"""
import torch, sys, time, gc, json, os
sys.argv = ["x"]
import fla  # register model types
from transformers import AutoModelForCausalLM

DEV = "cuda:0"
MODELS = {
    "Transformer": ("flash-linear-attention/flame/saves/Transformer-340M-10k", "full attention (O(L) cache)"),
    "SWA":         ("flash-linear-attention/flame/saves/SWA-340M-v2-10k",       "sliding window"),
    "GDN":         ("flash-linear-attention/flame/saves/GDN-340M-10k",          "gated DeltaNet (recurrent)"),
    "GMSWA":       ("flash-linear-attention/flame/saves/GMSWA-340M-v5conv-10k", "GM-SWA (ours)"),
}
LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
CHUNK = 8192          # chunked prefill granularity
DECODE_TOKS = 32      # tokens to average decode latency over

def chunked_prefill(m, L):
    pkv = None
    pos = 0
    while pos < L:
        n = min(CHUNK, L - pos)
        ids = torch.randint(0, 32000, (1, n), device=DEV)
        out = m(ids, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        pos += n
    return pkv, out.logits[:, -1:].argmax(-1)

def bench_one(m, L):
    torch.cuda.empty_cache(); gc.collect()
    base = torch.cuda.memory_allocated(DEV)
    # prefill (timed)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    pkv, nxt = chunked_prefill(m, L)
    torch.cuda.synchronize(); prefill_s = time.perf_counter() - t0
    total_mem = torch.cuda.memory_allocated(DEV) / 1e9            # model + cache (steady state)
    # decode latency: median over 3 timed bursts (robust to system hiccups)
    samples = []
    with torch.no_grad():
        for _rep in range(3):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            p2, n2 = pkv, nxt
            for _ in range(DECODE_TOKS):
                o = m(n2, past_key_values=p2, use_cache=True)
                p2 = o.past_key_values
                n2 = o.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize(); samples.append((time.perf_counter() - t0) * 1e3 / DECODE_TOKS)
            pkv, nxt = p2, n2
    samples.sort(); decode_ms = samples[1]
    prefill_tok_s = L / prefill_s
    del pkv, o
    return dict(decode_ms=round(decode_ms, 2), mem_gb=round(total_mem, 3),
                prefill_ktok_s=round(prefill_tok_s/1e3, 1))

def bench_model(name, path):
    print(f"\n##### {name} #####", flush=True)
    print(f"{'L':>8} {'decode_ms/tok':>14} {'mem_GB':>9} {'prefill_ktok/s':>15}", flush=True)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval()
    # warmup (compile kernels) at a small length, not timed
    with torch.no_grad():
        try: bench_one(m, 2048)
        except Exception: pass
    rows = {}
    for L in LENGTHS:
        try:
            with torch.no_grad():
                r = bench_one(m, L)
            print(f"{L:>8} {r['decode_ms']:>14.2f} {r['mem_gb']:>9.2f} {r['prefill_ktok_s']:>15.1f}", flush=True)
            rows[L] = r
        except torch.cuda.OutOfMemoryError:
            print(f"{L:>8} {'OOM':>14} {'OOM':>9} {'OOM':>15}", flush=True)
            rows[L] = "OOM"; torch.cuda.empty_cache()
        except Exception as e:
            print(f"{L:>8}  ERR {type(e).__name__}: {str(e)[:50]}", flush=True)
            rows[L] = "ERR"; torch.cuda.empty_cache()
    del m; gc.collect(); torch.cuda.empty_cache()
    return rows

if __name__ == "__main__":
    res = {name: bench_model(name, p) for name, (p, _) in MODELS.items()}
    os.makedirs("paper/figures", exist_ok=True)
    json.dump(res, open("paper/figures/efficiency.json", "w"), indent=2)
    print("\nRESULT_JSON " + json.dumps(res))
    # ---- figure ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        styles = {"Transformer": ("#d62728","o","-"), "SWA": ("#7f7f7f","s","--"),
                  "GDN": ("#2ca02c","^",":"), "GMSWA": ("#1f77b4","D","-")}
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        panels = [("decode_ms", "Decode latency (ms / token)", axes[0]),
                  ("mem_gb",    "Inference memory: model + KV cache (GB)", axes[1])]
        for key, ylab, ax in panels:
            for name, rows in res.items():
                xs = [L for L in LENGTHS if isinstance(rows.get(L), dict)]
                ys = [rows[L][key] for L in xs]
                if xs:
                    c, mk, ls = styles[name]
                    ax.plot(xs, ys, marker=mk, ls=ls, color=c, label=name, ms=5, lw=1.8)
            ax.set_xscale("log", base=2); ax.set_xlabel("context length (tokens)")
            ax.set_ylabel(ylab); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        axes[1].set_yscale("log")
        plt.tight_layout()
        plt.savefig("paper/figures/efficiency.pdf"); plt.savefig("paper/figures/efficiency.png", dpi=140)
        print("FIGURE_SAVED paper/figures/efficiency.{pdf,png}")
    except Exception as e:
        print("figure skipped:", type(e).__name__, str(e)[:80])
