"""Validate the v3 GM-SWA changes:
  1. model builds from the v3 config (separate mem proj, new gate init)
  2. mem_q_proj/mem_k_proj exist and gate biases are at the corrected init
  3. forward + backward run without NaN
  4. prefill-vs-incremental-decode CONSISTENCY (the new mem_k cache path is correct)
"""
import json, torch, fla  # noqa
from fla.models.gated_mem_swa import GatedMemSWAConfig, GatedMemSWAForCausalLM

cfg=GatedMemSWAConfig(**json.load(open("flash-linear-attention/flame/configs/gated_mem_swa_v3_340M.json")))
torch.manual_seed(0)
model=GatedMemSWAForCausalLM(cfg).cuda().to(torch.bfloat16).eval()
L0=model.model.layers[0].attn
print(f"mem_separate_proj={L0.mem_separate_proj} | mem_q_proj={tuple(L0.mem_q_proj.weight.shape)} mem_k_proj={tuple(L0.mem_k_proj.weight.shape)}")
import math
b=L0.gate_proj.bias.view(3,cfg.num_heads)
print(f"gate bias: write(beta) mean={b[0].mean():.2f} (sig={torch.sigmoid(b[0].mean()):.2f}), mix mean={b[2].mean():.2f} (alpha={torch.sigmoid(b[2].mean()):.2f})")
nparam=sum(p.numel() for p in model.parameters())/1e6
print(f"total params: {nparam:.1f}M")

V=cfg.vocab_size; N=600
ids=torch.randint(0,V,(1,N)).cuda()

# --- forward + backward (no cache) ---
model.train()
out=model(ids, labels=ids)
loss=out.loss
loss.backward()
gnorm=math.sqrt(sum((p.grad.float()**2).sum().item() for p in model.parameters() if p.grad is not None))
print(f"\nforward loss={loss.item():.3f}  nan={torch.isnan(loss).item()}  grad_norm={gnorm:.1f}  mem_q grad={'OK' if L0.mem_q_proj.weight.grad is not None and L0.mem_q_proj.weight.grad.abs().sum()>0 else 'ZERO!'}")
model.zero_grad(); model.eval()

# --- prefill vs incremental decode consistency ---
with torch.no_grad():
    full=model(ids, use_cache=False).logits[0,-1].float()
    # prefill first N-1, then decode last token with cache
    from fla.models.utils import Cache
    pkv=Cache.from_legacy_cache(None)
    o1=model(ids[:,:N-1], use_cache=True, past_key_values=pkv)
    pkv=o1.past_key_values
    o2=model(ids[:,N-1:N], use_cache=True, past_key_values=pkv)
    dec=o2.logits[0,-1].float()
cos=torch.nn.functional.cosine_similarity(full,dec,dim=0).item()
maxd=(full-dec).abs().max().item()
print(f"prefill-vs-decode: cosine={cos:.5f}  max|Δlogit|={maxd:.4f}  (should be ~1.0 / ~0)")
print("ARGMAX match:", (full.argmax()==dec.argmax()).item())
