"""Distance-swept recall probe with greedy generation, memory ON vs OFF.
Decides: does the memory ever recover cross-window recall, and is failure
distance-dependent (=> decay/capacity design issue) or total (=> wiring bug)?"""
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer
P="flash-linear-attention/flame/saves/GMSWA-340M-v2-10k"
tok=AutoTokenizer.from_pretrained(P)
model=AutoModelForCausalLM.from_pretrained(P,dtype=torch.bfloat16,trust_remote_code=True).cuda().eval()
layers=[l.attn for l in model.model.layers]
W=layers[0].window_size

filler="The garden was quiet and the wind moved slowly through the old trees. "  # ~14 tok/sentence
NEEDLE_VAL="8137"
def build(dist_tokens):
    """needle placed so it sits ~dist_tokens before the query end."""
    tail_sents=max(1,dist_tokens//14)
    pre = filler*8 + f"Important: the secret access token is {NEEDLE_VAL}. " + filler*tail_sents
    q   = "Question: the secret access token is"
    ids = tok(pre+q,return_tensors="pt").input_ids.cuda()
    # distance from needle phrase to end:
    return ids

def gen(ids):
    with torch.no_grad():
        out=model.generate(ids,max_new_tokens=6,do_sample=False,
                            pad_token_id=tok.eos_token_id,use_cache=True)
    return tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()

print(f"window={W}, needle='{NEEDLE_VAL}'")
print(f"{'approx_dist':>11} {'inlen':>6} {'mem_share':>9} | {'gen MEM-ON':>20} {'hit':>3} | {'gen MEM-OFF':>20} {'hit':>3}")
for dist in [200, 520, 800, 1400]:
    ids=build(dist)
    # runtime alpha
    al={}
    def hk(i,l):
        o=l._compute_gates
        def f(hs):
            g,b,m=o(hs); al[i]=(1-m.sigmoid().mean()).item(); return g,b,m
        return f
    hs_orig=[l._compute_gates for l in layers]
    for i,l in enumerate(layers): l._compute_gates=hk(i,l)
    g_on=gen(ids)
    for i,l in enumerate(layers): l._compute_gates=hs_orig[i]
    memshare=sum(al.values())/len(al)
    for l in layers: l.memory_enabled=False
    g_off=gen(ids)
    for l in layers: l.memory_enabled=True
    in_window = ids.shape[1] - 0  # rough
    hit_on = "Y" if NEEDLE_VAL in g_on else "."
    hit_off= "Y" if NEEDLE_VAL in g_off else "."
    print(f"{dist:>11} {ids.shape[1]:>6} {memshare:>9.3f} | {g_on[:20]:>20} {hit_on:>3} | {g_off[:20]:>20} {hit_off:>3}")
