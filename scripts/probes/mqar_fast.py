"""Fast, correct MQAR harness. Validates delta (fla kernel) reaches ~1.0, then
compares constant-memory mechanisms at EQUAL state (~4096 floats):
  delta-h4  : fla DeltaNet, 4 heads x dh=32  (rank cap 128)
  delta-h8  : fla DeltaNet, 8 heads x dh=16  (rank cap 128, smaller dh)
  hash      : hard-LSH bucket memory (L hashes, m buckets, dv values) -- escapes rank wall
  attn      : full attention (upper bound)
Usage: python mqar_fast.py <delta4|delta8|hash|attn>"""
import sys, math, torch, torch.nn as nn, torch.nn.functional as F, fla
from fla.ops.delta_rule import chunk_delta_rule
MIX=sys.argv[1]; dev="cuda"; torch.manual_seed(0)
V, DM, NL = 256, 128, 2

def gen(B,D):
    keys=torch.stack([torch.randperm(V,device=dev)[:D] for _ in range(B)]); vals=torch.randint(0,V,(B,D),device=dev)
    ctx=torch.stack([keys,vals],-1).view(B,2*D)
    perm=torch.stack([torch.randperm(D,device=dev) for _ in range(B)])
    qry=torch.stack([torch.gather(keys,1,perm),torch.gather(vals,1,perm)],-1).view(B,2*D)
    seq=torch.cat([ctx,qry],1); inp,tgt=seq[:,:-1].contiguous(),seq[:,1:].clone()
    msk=torch.zeros_like(tgt,dtype=torch.bool); msk[:,torch.arange(2*D,4*D-1,2,device=dev)]=True
    tgt[~msk]=-100; return inp,tgt

class Delta(nn.Module):
    def __init__(s,d,H,dh):
        super().__init__(); s.H=H; s.dh=dh
        s.q=nn.Linear(d,H*dh); s.k=nn.Linear(d,H*dh); s.v=nn.Linear(d,H*dh); s.b=nn.Linear(d,H); s.o=nn.Linear(H*dh,d)
    def forward(s,x):
        B,T,_=x.shape
        q=s.q(x).view(B,T,s.H,s.dh); k=s.k(x).view(B,T,s.H,s.dh); v=s.v(x).view(B,T,s.H,s.dh); beta=s.b(x).sigmoid()
        o,_=chunk_delta_rule(q.bfloat16(),k.bfloat16(),v.bfloat16(),beta.bfloat16(),
                             use_qk_l2norm_in_kernel=True,head_first=False)
        return s.o(o.float().reshape(B,T,-1))

class Hash(nn.Module):   # learnable, temperature-annealed soft->hard routing
    def __init__(s,d,L,m,dv,dk=32):
        super().__init__(); s.L=L; s.m=m; s.tau=1.0   # tau set/annealed by trainer
        s.rt=nn.Linear(d,dk); s.v=nn.Linear(d,dv); s.o=nn.Linear(dv,d)   # SHARED routing rt
        s.P=nn.Parameter(torch.randn(L,dk,m)/dk**0.5)          # LEARNABLE hash projections
    def assign(s,x):
        lg=torch.einsum('btd,ldm->btlm',F.normalize(s.rt(x),dim=-1),s.P)/max(s.tau,1e-3)
        soft=lg.softmax(-1)
        if s.tau<=0.06:                                        # straight-through once sharp
            idx=lg.argmax(-1,keepdim=True); hard=torch.zeros_like(soft).scatter_(-1,idx,1.0)
            return hard+soft-soft.detach()
        return soft
    def forward(s,x):
        B,T,_=x.shape; A=s.assign(x); v=s.v(x)                 # same routing for read & write
        match=torch.einsum('btlm,bilm->bti',A,A)
        match=match*torch.tril(torch.ones(T,T,device=x.device),-1)
        match=match/(match.sum(-1,keepdim=True)+1e-6)
        return s.o(match@v)

class Attn(nn.Module):
    def __init__(s,d):
        super().__init__(); s.q=nn.Linear(d,d); s.k=nn.Linear(d,d); s.v=nn.Linear(d,d); s.o=nn.Linear(d,d)
    def forward(s,x):
        q,k,v=s.q(x),s.k(x),s.v(x)
        return s.o(F.scaled_dot_product_attention(q.unsqueeze(1),k.unsqueeze(1),v.unsqueeze(1),is_causal=True).squeeze(1))

class PKMem(nn.Module):   # learnable codebook + differentiable TOP-K softmax routing (PKM-style)
    def __init__(s,d,m,dv,ksel=4,dk=32):
        super().__init__(); s.m=m; s.ksel=ksel
        s.rt=nn.Linear(d,dk); s.v=nn.Linear(d,dv); s.o=nn.Linear(dv,d)
        s.C=nn.Parameter(F.normalize(torch.randn(m,dk),dim=-1))   # learnable code/bucket keys
    def assign(s,x):
        sc=F.normalize(s.rt(x),dim=-1)@F.normalize(s.C,dim=-1).T   # (B,T,m) cosine
        topv,topi=sc.topk(s.ksel,-1); w=(topv*8.0).softmax(-1)     # soft over the top-k selected codes
        A=torch.zeros_like(sc).scatter(-1,topi,w)                  # sparse (ksel active), differentiable
        return A
    def forward(s,x):
        B,T,_=x.shape; A=s.assign(x); v=s.v(x)
        match=torch.einsum('btm,bim->bti',A,A)*torch.tril(torch.ones(T,T,device=x.device),-1)
        match=match/(match.sum(-1,keepdim=True)+1e-6)
        return s.o(match@v)

class Linear(nn.Module):   # linear attention (elu+1 feature), causal normalized — the real baseline
    def __init__(s,d):
        super().__init__(); s.q=nn.Linear(d,d); s.k=nn.Linear(d,d); s.v=nn.Linear(d,d); s.o=nn.Linear(d,d)
    def forward(s,x):
        B,T,_=x.shape; q=F.elu(s.q(x))+1; k=F.elu(s.k(x))+1; v=s.v(x)
        A=torch.tril(q@k.transpose(1,2)); A=A/(A.sum(-1,keepdim=True)+1e-6)
        return s.o(A@v)

class ShortConv(nn.Module):  # causal depthwise conv — lets mixers form adjacent-token associations
    def __init__(s,d,k=4):
        super().__init__(); s.k=k; s.conv=nn.Conv1d(d,d,k,groups=d,padding=k-1)
    def forward(s,x):
        return F.silu(s.conv(x.transpose(1,2))[...,:x.shape[1]].transpose(1,2))

def mk():
    return {"delta4":lambda:Delta(DM,4,32),"delta8":lambda:Delta(DM,8,16),
            "hash":lambda:Hash(DM,1,256,16),"pkm":lambda:PKMem(DM,256,16,ksel=4),
            "linear":lambda:Linear(DM),"attn":lambda:Attn(DM)}[MIX]()
class LM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb=nn.Embedding(V,DM); s.pos=nn.Embedding(2048,DM)
        s.mix=nn.ModuleList([mk() for _ in range(NL)]); s.n1=nn.ModuleList([nn.LayerNorm(DM) for _ in range(NL)])
        s.sc=nn.ModuleList([ShortConv(DM) for _ in range(NL)])
        s.n2=nn.ModuleList([nn.LayerNorm(DM) for _ in range(NL)])
        s.mlp=nn.ModuleList([nn.Sequential(nn.Linear(DM,4*DM),nn.GELU(),nn.Linear(4*DM,DM)) for _ in range(NL)])
        s.nf=nn.LayerNorm(DM); s.head=nn.Linear(DM,V)
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device))
        for i in range(NL): h=h+s.mix[i](s.sc[i](s.n1[i](h))); h=h+s.mlp[i](s.n2[i](h))
        return s.head(s.nf(h))

def run(D,steps=12000,B=256,peak=3e-3,warm=800):
    torch.manual_seed(0); m=LM().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=peak,weight_decay=0.01,betas=(0.9,0.98))
    hms=[mod for mod in m.modules() if isinstance(mod,Hash)]
    for it in range(steps):
        lr=peak*it/warm if it<warm else peak*0.5*(1+math.cos(math.pi*(it-warm)/(steps-warm)))
        for g in opt.param_groups: g['lr']=lr
        tau=max(0.05, 1.0*(0.05**(min(it,int(steps*0.8))/(steps*0.8))))   # anneal 1.0->0.05 over first 80%
        for hm in hms: hm.tau=tau
        inp,tgt=gen(B,D); loss=F.cross_entropy(m(inp).reshape(-1,V),tgt.reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    for hm in hms: hm.tau=0.05    # sharp (hard) routing at eval
    m.eval(); c=t=0
    with torch.no_grad():
        for _ in range(20):
            inp,tgt=gen(256,D); p=m(inp).argmax(-1); msk=tgt!=-100; c+=(p[msk]==tgt[msk]).sum().item(); t+=msk.sum().item()
    return c/t

print(f"=== MQAR-fast mixer={MIX} (state~4096, d_model=128, 12k steps) ===",flush=True)
for D in [16,32,64,96]:
    print(f"  D={D:>3} (L={4*D:>3}): acc={run(D):.3f}",flush=True)
