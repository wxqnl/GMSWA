"""No-training toy: what is the EXACT-recall ceiling of different memory readouts?
Plant N key->value pairs (random), query one planted key, measure how well each
readout reconstructs that key's value (cosine to ground truth). This bounds what
each mechanism CAN do before we spend 5h training it.

Mechanisms:
  A) delta-rule state (sum of beta * v k^T, L2-normed k) read by q   [v3's memory]
  B) full softmax over all tokens (= full attention; upper bound)
  C) chunked softmax, mean-pooled K & V, chunk size C (compressive)  [candidate]
  D) chunked softmax for SELECTION + exact value of best-matching token in block
"""
import torch, torch.nn.functional as F
torch.manual_seed(0)
d=64; N=512; C=4
def trial(N, query_pos):
    K=F.normalize(torch.randn(N,d),dim=-1)   # keys
    V=torch.randn(N,d)                         # values
    q=K[query_pos].clone()                     # query = exact key of the needle (ideal retrieval cue)
    vtrue=V[query_pos]
    cos=lambda a,b: F.cosine_similarity(a.reshape(-1),b.reshape(-1),dim=0).item()
    # A) delta-ish: state = sum_i v_i k_i^T ; read o = state q = sum_i v_i (k_i.q)
    o_delta = (V * (K@q).unsqueeze(-1)).sum(0)
    # B) full softmax
    a=( (K@q)* (d**0.5) ).softmax(0); o_full=(a.unsqueeze(-1)*V).sum(0)
    # C) chunked mean-pool softmax
    nb=N//C
    Ksum=K.view(nb,C,d).mean(1); Vsum=V.view(nb,C,d).mean(1)
    Ksum=F.normalize(Ksum,dim=-1)
    ac=((Ksum@q)*(d**0.5)).softmax(0); o_chunk=(ac.unsqueeze(-1)*Vsum).sum(0)
    # D) chunk-select then exact best token in selected block
    blk=int((Ksum@q).argmax()); seg=slice(blk*C,(blk+1)*C)
    sel=((K[seg]@q)*(d**0.5)).softmax(0); o_sel=(sel.unsqueeze(-1)*V[seg]).sum(0)
    return cos(o_delta,vtrue),cos(o_full,vtrue),cos(o_chunk,vtrue),cos(o_sel,vtrue)
import numpy as np
res=np.array([trial(N, p) for p in range(0,N,7)])
m=res.mean(0)
print(f"recall (cosine to true value) over {len(res)} planted needles, N={N} tokens, chunk C={C}:")
print(f"  A) delta-rule readout (v3):           {m[0]:.3f}")
print(f"  B) full softmax (upper bound):        {m[1]:.3f}")
print(f"  C) chunked mean-pool softmax (C={C}):   {m[2]:.3f}")
print(f"  D) chunk-select + exact token:        {m[3]:.3f}")
