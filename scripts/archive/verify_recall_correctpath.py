"""Re-measure trained-baseline out-of-window recall via the CORRECT dense path
(a single full forward = exactly the prefill path, no buggy incremental decode).
Decides whether the earlier generate()-based 'memory fails' conclusion was
confounded by the decode bug."""
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer
P="flash-linear-attention/flame/saves/GMSWA-340M-v2-10k"
tok=AutoTokenizer.from_pretrained(P)
model=AutoModelForCausalLM.from_pretrained(P,dtype=torch.bfloat16,trust_remote_code=True).cuda().eval()
W=model.model.layers[0].attn.window_size

filler="The garden was quiet and the wind moved slowly through the old trees. "
NEEDLE="8137"
def build(dist):
    pre=filler*8+f"Important: the secret access token is {NEEDLE}. "+filler*max(1,dist//14)
    return tok(pre+"Question: the secret access token is",return_tensors="pt").input_ids.cuda()

# needle content token ids (no leading space, and with space)
cand=set()
for s in (NEEDLE," "+NEEDLE):
    ids=tok(s,add_special_tokens=False).input_ids
    cand.add(ids[0])
cand=list(cand)
print(f"window={W}, needle={NEEDLE}, first-content-token candidates={[(c,tok.decode([c])) for c in cand]}")
for dist in [200, 700, 1400]:
    ids=build(dist)
    with torch.no_grad():
        lg=model(ids).logits[0,-1].float()
    probs=lg.softmax(0)
    best_c=max(cand,key=lambda c: probs[c].item())
    rank=int((lg>lg[best_c]).sum())
    top=[(tok.decode([j]),round(v.item(),2)) for v,j in zip(*lg.topk(8))]
    inwin = "IN-WIN" if ids.shape[1]<W else "OUT"
    print(f"dist~{dist:>4} len={ids.shape[1]:>4} [{inwin:6s}] P(needle-tok)={probs[best_c].item():.4f} rank={rank:>5} | top8={top}")
