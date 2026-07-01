"""Mechanistic probe: is the GM-SWA memory branch doing anything at inference?
Builds a cross-window recall input (needle ~1000 tokens before the query, far
outside the 512 window) and measures (1) runtime mix-gate alpha per layer,
(2) whether disabling the memory changes the logits / recall at all."""
import torch, fla  # noqa: F401  (registers gated_mem_swa)
from transformers import AutoModelForCausalLM, AutoTokenizer

P="flash-linear-attention/flame/saves/GMSWA-340M-v2-10k"
tok=AutoTokenizer.from_pretrained(P)
model=AutoModelForCausalLM.from_pretrained(P,torch_dtype=torch.bfloat16,trust_remote_code=True).cuda().eval()

# locate attn layers
layers=[l.attn for l in model.model.layers]
print(f"{len(layers)} attn layers, window={layers[0].window_size}")

# --- build cross-window recall input ---
needle_val="8137"
filler="The garden was quiet and the wind moved slowly through the old trees. "
# needle early, then lots of filler so needle is far outside the 512 window
ctx = filler*12 + f"Important: the secret access token is {needle_val}. " + filler*120 \
      + "Question: the secret access token is"
ids=tok(ctx,return_tensors="pt").input_ids.cuda()
val_ids=tok(" "+needle_val,add_special_tokens=False).input_ids
print(f"input len={ids.shape[1]} tokens; needle '{needle_val}' first-tok id={val_ids[0]} ({tok.decode(val_ids[:1])!r})")

# --- capture runtime alpha (=SWA share) per layer ---
alphas={}; betas={}
def hook(i,layer):
    orig=layer._compute_gates
    def f(hs):
        g,beta,mix=orig(hs)
        alphas[i]=mix.sigmoid().mean().item(); betas[i]=beta.mean().item()
        return g,beta,mix
    return f
for i,l in enumerate(layers): l._compute_gates=hook(i,l)

def run():
    with torch.no_grad():
        return model(ids).logits[0,-1].float()

logits_on=run()
a=sum(alphas.values())/len(alphas); b=sum(betas.values())/len(betas)
print(f"\nRUNTIME on real cross-window input: mean alpha(SWA share)={a:.3f} -> memory share={1-a:.3f} | mean write beta={b:.3f}")
print(f"per-layer memory share (1-alpha): "+" ".join(f"{1-alphas[i]:.2f}" for i in range(len(layers))))

# --- ablation: disable memory entirely, same weights ---
for l in layers: l.memory_enabled=False
logits_off=run()
for l in layers: l.memory_enabled=True

import torch.nn.functional as F
cos=F.cosine_similarity(logits_on,logits_off,dim=0).item()
maxd=(logits_on-logits_off).abs().max().item()
def topk(lg,k=5):
    v,idx=lg.topk(k); return [(tok.decode([j]),round(x.item(),2)) for x,j in zip(v,idx)]
rank_on=(logits_on>logits_on[val_ids[0]]).sum().item()
rank_off=(logits_off>logits_off[val_ids[0]]).sum().item()
print(f"\n--- ABLATION (memory ON vs OFF, same model) ---")
print(f"cosine(logits_on, logits_off) = {cos:.5f}   max|Δlogit| = {maxd:.3f}")
print(f"P(correct needle tok): on={logits_on.softmax(0)[val_ids[0]].item():.4f}  off={logits_off.softmax(0)[val_ids[0]].item():.4f}")
print(f"rank of needle tok:    on={rank_on}  off={rank_off}  (0=top; lower=better recall)")
print(f"top-5 ON : {topk(logits_on)}")
print(f"top-5 OFF: {topk(logits_off)}")
