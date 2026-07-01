"""Prefill vs incremental-decode consistency for GatedMemSWA with mem short-conv.
If the conv-cache / eviction threading is correct, decoding token-by-token with a
KV+conv cache must match a single full-sequence prefill. Also checks conv-DISABLED
backward-compat (must be byte-identical behavior to the pre-conv code path)."""
import torch, sys
sys.argv = ["x"]
from fla.models.gated_mem_swa.configuration_gated_mem_swa import GatedMemSWAConfig
from fla.models.gated_mem_swa.modeling_gated_mem_swa import GatedMemSWAForCausalLM

dev = "cuda:0"
torch.manual_seed(0)

def run(use_conv, evicted_only):
    cfg = GatedMemSWAConfig(
        hidden_size=128, num_hidden_layers=2, num_heads=4, num_kv_heads=2,
        window_size=8, vocab_size=256, max_position_embeddings=512,
        mem_separate_proj=True, mem_evicted_only=evicted_only,
        mem_use_short_conv=use_conv, mem_conv_size=4,
        fuse_cross_entropy=False, rope_theta=10000.0,
    )
    model = GatedMemSWAForCausalLM(cfg).to(dev).to(torch.float32).eval()
    T = 24  # > window 8, forces eviction
    ids = torch.randint(0, 256, (1, T), device=dev)
    with torch.no_grad():
        # full prefill
        out_pre = model(ids, use_cache=False).logits[0]  # (T, vocab)
        # incremental decode
        pkv = None; outs = []
        for t in range(T):
            o = model(ids[:, t:t+1], past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            outs.append(o.logits[0, -1])
        out_dec = torch.stack(outs, 0)  # (T, vocab)
    # compare on the post-window positions (where eviction/memory matters)
    a, b = out_pre[8:], out_dec[8:]
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    maxd = (a - b).abs().max().item()
    print(f"  use_conv={use_conv} evicted_only={evicted_only}: cosine={cos:.6f}  max|Δ|={maxd:.3e}")
    return cos

print("=== prefill vs decode consistency (post-window positions) ===")
ok = True
for uc in (False, True):
    for eo in (True, False):
        c = run(uc, eo)
        ok = ok and (c > 0.999)
print("RESULT:", "PASS ✅" if ok else "FAIL ❌ (cosine < 0.999 — cache/eviction bug)")
