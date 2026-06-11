"""Pure mechanism capacity — NO training, no optimization confounds.
Store D (key->token) associations in a fixed-state memory, query each key, decode
the retrieved value-code to a token via a shared codebook. Measure exact-token
recall vs D, at EQUAL state size. Tests the core theory:
  - delta/linear (bilinear): capacity = rank = min(d_k,d_v); D assoc need ~D^2 state
  - hash table: capacity = #buckets, DECOUPLED from value-dim; D assoc need ~D state
"""
import torch, torch.nn.functional as F, numpy as np
dev="cuda"; torch.manual_seed(0)
V=256; STATE=4096; TRIALS=200

def trial(D, mech, d_k=64, d_v=64, m=256, L=1):
    cb=F.normalize(torch.randn(V,d_v,device=dev),dim=-1)              # token codebook
    K=F.normalize(torch.randn(D,d_k,device=dev),dim=-1)              # keys
    t=torch.randint(0,V,(D,),device=dev); Vv=cb[t]                   # values = token codes
    if mech in ("linear","delta"):
        S=torch.zeros(d_v,d_k,device=dev)
        if mech=="linear":
            S=Vv.T@K                                                  # sum of outer products
        else:                                                         # sequential delta rule
            for i in range(D):
                ki=K[i]; pred=S@ki; S=S+torch.outer(Vv[i]-pred,ki)
        o=(S@K.T).T                                                   # read each key (D,d_v)
    elif mech=="hash":
        P=torch.randn(L,d_k,m,device=dev)/d_k**0.5
        buck=(K@P).argmax(-1)                                         # (L,D) bucket per hash
        M=torch.zeros(L,m,d_v,device=dev); cnt=torch.zeros(L,m,device=dev)
        for l in range(L):
            M[l].index_add_(0,buck[l],Vv); cnt[l].index_add_(0,buck[l],torch.ones(D,device=dev))
        os=[]
        for l in range(L):
            b=buck[l]; os.append(M[l][b]/cnt[l][b].clamp(min=1).unsqueeze(-1))
        o=torch.stack(os).mean(0)                                     # (D,d_v) aggregate hashes
    pred=(o@cb.T).argmax(-1)                                          # decode to token
    return (pred==t).float().mean().item()

def sweep(mech,**kw):
    return {D:np.mean([trial(D,mech,**kw) for _ in range(TRIALS//4)]) for D in [16,32,48,64,96,128,192]}

print(f"=== exact-token recall vs #associations D (state={STATE}) ===")
print(f"{'D':>5} {'linear':>8} {'delta':>8} {'hash m256 dv16':>15} {'hash m512 dv8':>14} {'hash m1024 dv4':>15}")
r_lin=sweep("linear"); r_del=sweep("delta")
r_h256=sweep("hash",d_k=64,m=256,d_v=16,L=1)      # state 256*16=4096
r_h512=sweep("hash",d_k=64,m=512,d_v=8,L=1)       # 512*8=4096
r_h1024=sweep("hash",d_k=64,m=1024,d_v=4,L=1)     # 1024*4=4096
for D in [16,32,48,64,96,128,192]:
    print(f"{D:>5} {r_lin[D]:>8.3f} {r_del[D]:>8.3f} {r_h256[D]:>15.3f} {r_h512[D]:>14.3f} {r_h1024[D]:>15.3f}")
