"""De-risk experiment: does a fixed-K-slot content-addressable softmax memory
recall better than a DeltaNet matrix state at EQUAL state size, on MQAR?

Same tiny backbone (embed + 2 layers + MLP head); only the sequence mixer differs.
Sweep number of key-value pairs D at fixed matched state. Report recall accuracy.

Usage: python mqar_derisk.py <kslot|delta|attn>
State match @ d_model=64: delta state = 64x64 = 4096 floats; kslot K=32,d=64 -> 2*32*64 = 4096.
"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
MIX = sys.argv[1]
dev = "cuda"
torch.manual_seed(0)
V, DMODEL, NLAYER, KSLOTS = 256, 64, 2, 32

def gen_mqar(B, D):
    keys = torch.stack([torch.randperm(V, device=dev)[:D] for _ in range(B)])     # distinct keys
    vals = torch.randint(0, V, (B, D), device=dev)
    ctx = torch.stack([keys, vals], -1).view(B, 2*D)
    perm = torch.stack([torch.randperm(D, device=dev) for _ in range(B)])
    qk = torch.gather(keys, 1, perm); qv = torch.gather(vals, 1, perm)
    qry = torch.stack([qk, qv], -1).view(B, 2*D)
    seq = torch.cat([ctx, qry], 1)                                                 # (B,4D)
    inp, tgt = seq[:, :-1].contiguous(), seq[:, 1:].clone()
    mask = torch.zeros_like(tgt, dtype=torch.bool)
    mask[:, torch.arange(2*D, 4*D-1, 2, device=dev)] = True                        # answer-pred positions
    tgt[~mask] = -100
    return inp, tgt

class KSlot(nn.Module):
    def __init__(s, d, K):
        super().__init__(); s.K=K
        s.q=nn.Linear(d,d); s.k=nn.Linear(d,d); s.v=nn.Linear(d,d); s.b=nn.Linear(d,1); s.o=nn.Linear(d,d)
        s.Sk0=nn.Parameter(torch.randn(K,d)*0.02)
    def forward(s,x):
        B,T,d=x.shape; sc=d**-0.5
        q=s.q(x); k=s.k(x); v=s.v(x); beta=s.b(x).sigmoid()
        Sk=s.Sk0.unsqueeze(0).expand(B,-1,-1).contiguous(); Sv=torch.zeros(B,s.K,d,device=x.device)
        outs=[]
        for t in range(T):
            r=(q[:,t:t+1]@Sk.transpose(1,2)*sc).softmax(-1)        # (B,1,K)
            outs.append((r@Sv).squeeze(1))                          # (B,d)
            w=(k[:,t:t+1]@Sk.transpose(1,2)*sc).softmax(-1)         # (B,1,K)
            bw=(beta[:,t:t+1].transpose(1,2)*w).transpose(1,2)      # (B,K,1)
            Sv=Sv+bw*(v[:,t:t+1]-Sv); Sk=Sk+bw*(k[:,t:t+1]-Sk)
        return s.o(torch.stack(outs,1))

class Delta(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.q=nn.Linear(d,d); s.k=nn.Linear(d,d); s.v=nn.Linear(d,d); s.b=nn.Linear(d,1); s.o=nn.Linear(d,d)
    def forward(s,x):
        B,T,d=x.shape
        q=s.q(x); k=F.normalize(s.k(x),dim=-1); v=s.v(x); beta=s.b(x).sigmoid()
        S=torch.zeros(B,d,d,device=x.device); outs=[]
        for t in range(T):
            outs.append((S@q[:,t].unsqueeze(-1)).squeeze(-1))
            kt=k[:,t]; Sk=(S@kt.unsqueeze(-1)).squeeze(-1)
            S=S+(beta[:,t]*(v[:,t]-Sk)).unsqueeze(-1)*kt.unsqueeze(1)
        return s.o(torch.stack(outs,1))

class Attn(nn.Module):
    def __init__(s,d):
        super().__init__(); s.q=nn.Linear(d,d); s.k=nn.Linear(d,d); s.v=nn.Linear(d,d); s.o=nn.Linear(d,d)
    def forward(s,x):
        B,T,d=x.shape
        q,k,v=s.q(x),s.k(x),s.v(x)
        o=F.scaled_dot_product_attention(q.unsqueeze(1),k.unsqueeze(1),v.unsqueeze(1),is_causal=True).squeeze(1)
        return s.o(o)

def mk_mixer():
    return {"kslot":lambda:KSlot(DMODEL,KSLOTS),"delta":lambda:Delta(DMODEL),"attn":lambda:Attn(DMODEL)}[MIX]()

class TinyLM(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb=nn.Embedding(V,DMODEL); s.pos=nn.Embedding(2048,DMODEL)
        s.mix=nn.ModuleList([mk_mixer() for _ in range(NLAYER)])
        s.mn=nn.ModuleList([nn.LayerNorm(DMODEL) for _ in range(NLAYER)])
        s.fn=nn.ModuleList([nn.LayerNorm(DMODEL) for _ in range(NLAYER)])
        s.mlp=nn.ModuleList([nn.Sequential(nn.Linear(DMODEL,4*DMODEL),nn.GELU(),nn.Linear(4*DMODEL,DMODEL)) for _ in range(NLAYER)])
        s.norm=nn.LayerNorm(DMODEL); s.head=nn.Linear(DMODEL,V)
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device))
        for i in range(NLAYER):
            h=h+s.mix[i](s.mn[i](h)); h=h+s.mlp[i](s.fn[i](h))
        return s.head(s.norm(h))

def run(D, steps=2500, B=128):
    torch.manual_seed(0)
    m=TinyLM().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=0.01)
    for it in range(steps):
        inp,tgt=gen_mqar(B,D)
        loss=F.cross_entropy(m(inp).reshape(-1,V),tgt.reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    # eval accuracy on answer positions
    m.eval(); correct=tot=0
    with torch.no_grad():
        for _ in range(20):
            inp,tgt=gen_mqar(256,D); pred=m(inp).argmax(-1)
            msk=tgt!=-100; correct+=(pred[msk]==tgt[msk]).sum().item(); tot+=msk.sum().item()
    return correct/tot

print(f"=== MQAR  mixer={MIX}  (state ~4096 floats: delta 64x64, kslot K=32xd64) ===", flush=True)
state = "4096" if MIX!="attn" else "GROWS (upper bound)"
print(f"state size: {state}", flush=True)
for D in [8,16,24,32,48]:
    acc=run(D)
    print(f"  D={D:>3} pairs (seq_len={4*D:>3}): recall_acc={acc:.3f}", flush=True)
