"""Long-context inference efficiency: prefill latency, per-token decode latency,
and peak GPU memory vs sequence length, for the 4 architectures. Shows GMSWA's
constant KV-cache advantage (decode latency + memory flat in L) vs full attention
(both grow with L). Single GPU, bf16."""
import torch, sys, time, gc
sys.argv = ["x"]
import fla  # register model types
from transformers import AutoModelForCausalLM

DEV = "cuda:0"
MODELS = {
    "Transformer": "flash-linear-attention/flame/saves/Transformer-340M-10k",  # full attn O(L) cache
    "SWA":         "flash-linear-attention/flame/saves/SWA-340M-v2-10k",        # window, constant cache
    "GDN":         "flash-linear-attention/flame/saves/GDN-340M-10k",           # recurrent, constant state
    "GMSWA":       "flash-linear-attention/flame/saves/GMSWA-340M-v5conv-10k",  # SWA + memory, constant cache
}
LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
DECODE_TOKS = 16

def bench(name, path):
    print(f"\n##### {name} #####", flush=True)
    print(f"{'L':>7} {'prefill_ms':>11} {'decode_ms/tok':>14} {'peak_mem_GB':>12}", flush=True)
    rows = []
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval()
    for L in LENGTHS:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(DEV)
        try:
            ids = torch.randint(0, 32000, (1, L), device=DEV)
            with torch.no_grad():
                # prefill
                torch.cuda.synchronize(); t0 = time.perf_counter()
                out = m(ids, use_cache=True)
                torch.cuda.synchronize(); prefill_ms = (time.perf_counter() - t0) * 1e3
                pkv = out.past_key_values
                nxt = ids[:, -1:]
                # decode
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(DECODE_TOKS):
                    o = m(nxt, past_key_values=pkv, use_cache=True)
                    pkv = o.past_key_values
                    nxt = o.logits[:, -1:].argmax(-1)
                torch.cuda.synchronize(); dec_ms = (time.perf_counter() - t0) * 1e3 / DECODE_TOKS
            peak = torch.cuda.max_memory_allocated(DEV) / 1e9
            print(f"{L:>7} {prefill_ms:>11.1f} {dec_ms:>14.2f} {peak:>12.2f}", flush=True)
            rows.append((L, round(prefill_ms,1), round(dec_ms,2), round(peak,2)))
            del out, pkv, o
        except torch.cuda.OutOfMemoryError:
            print(f"{L:>7} {'OOM':>11} {'OOM':>14} {'OOM':>12}", flush=True)
            rows.append((L, "OOM", "OOM", "OOM"))
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{L:>7}  ERROR {type(e).__name__}: {str(e)[:60]}", flush=True)
            rows.append((L, "ERR", "ERR", "ERR"))
            torch.cuda.empty_cache()
    del m; gc.collect(); torch.cuda.empty_cache()
    return rows

if __name__ == "__main__":
    import json
    allr = {}
    # warm up triton kernels on a tiny run is implicit in first L
    for name, path in MODELS.items():
        allr[name] = bench(name, path)
    print("\nRESULT_JSON " + json.dumps(allr))
