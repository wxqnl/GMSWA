"""MQAR de-risk v2: fixed optimization (warmup+cosine) + sharpened K-slot write.
Tests whether a *better* K-slot content-addressable softmax memory can match
DeltaNet at equal constant state. Usage: python mqar_derisk2.py <kslot|delta|attn>"""
import sys, math, torch, torch.nn as nn, torch.nn.functional as F
MIX = sys.argv[1]; dev="cuda"; torch.manual_seed(0)
V, DMODEL, NLAYER, KSLOTS = 256, 64, 2, 32

def gen_mqar(B, D):
    keys=torch.stack([torch.randperm(V,device=dev)[:D] for _ in range(B)])
    vals=torch.randint(0,V,(B,D),device=dev)
    ctx=torch.stack([keys,vals],-1).view(B,2*D)
    perm=torch.stack([torch.randperm(D,device=dev) for _ in range(B)])
    qry=torch.stack([torch.gather(keys,1,perm),torch.gather(vals,1,perm)],-1).view(B,2*D)
    seq=torch.cat([ctx,qry],1); inp,tgt=seq[:,:-1].contiguous(),seq[:,1:].clone()
    msk=torch.zeros_like(tgt,dtype=torch.bool); msk[:,torch.arange(2*D,4*D-1,2,device=dev)]=True
    tgt[~msk]=-100; return inp,tgt

class KSlot(nn.Module):  # v2: cosine addressing + learnable sharpness
    def __init__(s,d,K):
        super().__init__(); s.K=K
        s.q=nn.Linear(d,d); s.k=nn.Linear(d,d); s.v=nn.Linear(d,d); s.b=nn.Linear(d,1); s.o=nn.Linear(d,d)
        s.Sk0=nn.Parameter(F.normalize(torch.randn(K,d),dim=-1)*1.0)
        s.tmp=nn.Parameter(torch.tensor(8.0))   # addressing sharpness (learnable)
    def forward(s,x):
        B,T,d=x.shape
        q=s.q(x); k=s.k(x); v=s.v(x); beta=s.b(x).sigmoid()
        Sk=s.Sk0.unsqueeze(0).expand(B,-1,-1).contiguous(); Sv=torch.zeros(B,s.K,d,device=x.device)
        outs=[]; tmp=F.softplus(s.tmp)
        for t in range(T):
            qn=F.normalize(q[:,t:t+1],dim=-1); kn=F.normalize(k[:,t:t+1],dim=-1); Skn=F.normalize(Sk,dim=-1)
            r=(qn@Skn.transpose(1,2)*tmp).softmax(-1)        # (B,1,K) sharp read
            outs.append((r@Sv).squeeze(1))
            w=(kn@Skn.transpose(1,2)*tmp).softmax(-1)        # (B,1,K) sharp write
            bw=(beta[:,t:t+1].transpose(1,2)*w).transpose(1,2)
            Sv=Sv+bw*(v[:,t:t+1]-Sv); Sk=Sk+bw*(k[:,t:t+1]-Sk)
        return s.o(torch.stack(outs,1))

class Delta(nn.Module):
    def __init__(s,d):
        super().__init__(); s.q=nn.Linear(d,d); s.k=nn.Linear(d,d); s.v=nn.Linear(d,d); s.b=nn.Linear(d,1); s.o=nn.Linear(d,d)
    def forward(s,x):
        B,T,d=x.shape; q=s.q(x); k=F.normalize(s.k(x),dim=-1); v=s.v(x); beta=s.b(x).sigmoid()
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
        q,k,v=s.q(x),s.k(x),s.v(x)
        o=F.scaled_dot_product_attention(q.unsqueeze(1),k.unsqueeze(1),v.unsqueeze(1),is_causal=True).squeeze(1)
        return s.o(o)

def mk(): return {"kslot":lambda:KSlot(DMODEL,KSLOTS),"delta":lambda:Delta(DMODEL),"attn":lambda:Attn(DMODEL)}[MIX]()
class TinyLM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb=nn.Embedding(V,DMODEL); s.pos=nn.Embedding(2048,DMODEL)
        s.mix=nn.ModuleList([mk() for _ in range(NLAYER)]); s.mn=nn.ModuleList([nn.LayerNorm(DMODEL) for _ in range(NLAYER)])
        s.fn=nn.ModuleList([nn.LayerNorm(DMODEL) for _ in range(NLAYER)])
        s.mlp=nn.ModuleList([nn.Sequential(nn.Linear(DMODEL,4*DMODEL),nn.GELU(),nn.Linear(4*DMODEL,DMODEL)) for _ in range(NLAYER)])
        s.norm=nn.LayerNorm(DMODEL); s.head=nn.Linear(DMODEL,V)
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device))
        for i in range(NLAYER): h=h+s.mix[i](s.mn[i](h)); h=h+s.mlp[i](s.fn[i](h))
        return s.head(s.norm(h))

def run(D, steps=3500, B=128, peak=1.5e-3, warm=350):
    torch.manual_seed(0); m=TinyLM().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=peak,weight_decay=0.01,betas=(0.9,0.98))
    def lr_at(i): return peak*(i/warm) if i<warm else peak*0.5*(1+math.cos(math.pi*(i-warm)/(steps-warm)))
    for it in range(steps):
        for g in opt.param_groups: g['lr']=lr_at(it)
        inp,tgt=gen_mqar(B,D)
        loss=F.cross_entropy(m(inp).reshape(-1,V),tgt.reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval(); c=tot=0
    with torch.no_grad():
        for _ in range(20):
            inp,tgt=gen_mqar(256,D); p=m(inp).argmax(-1); msk=tgt!=-100; c+=(p[msk]==tgt[msk]).sum().item(); tot+=msk.sum().item()
    return c/tot

print(f"=== MQAR v2  mixer={MIX} (warmup+cosine; state 4096: delta 64x64=64 assoc, kslot K=32) ===",flush=True)
for D in [8,16,24,32,48]:
    print(f"  D={D:>3} (L={4*D:>3}): acc={run(D):.3f}",flush=True)
