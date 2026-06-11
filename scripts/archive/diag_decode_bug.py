"""Pinpoint the eviction decode bug. Tests, in order:
  A) reproduce: full(chunk) vs prefill(chunk)+decode(recurrent)
  B) is it the kernel family? force prefill to RECURRENT, re-test
  C) direct: chunk vs recurrent equivalence on the SAME delta-rule inputs
  D) handoff: recurrent[0:P]+recurrent[P:] vs recurrent[0:] (state carry correctness)
"""
import json, torch, types, fla  # noqa
from fla.models.gated_mem_swa import GatedMemSWAConfig, GatedMemSWAForCausalLM
from fla.models.utils import Cache
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

cfg=GatedMemSWAConfig(**json.load(open("flash-linear-attention/flame/configs/gated_mem_swa_v3_340M.json")))
torch.manual_seed(0)
m=GatedMemSWAForCausalLM(cfg).cuda().to(torch.bfloat16).eval()
N=600; ids=torch.randint(0,cfg.vocab_size,(1,N)).cuda()
def cos(a,b): return torch.nn.functional.cosine_similarity(a.float(),b.float(),dim=0).item()

def prefill_decode():
    with torch.no_grad():
        pkv=Cache.from_legacy_cache(None)
        m(ids[:,:N-1],use_cache=True,past_key_values=pkv)
        return m(ids[:,N-1:N],use_cache=True,past_key_values=pkv).logits[0,-1].float()
with torch.no_grad(): full=m(ids,use_cache=False).logits[0,-1].float()
print(f"[A] full(chunk) vs prefill(chunk)+decode(rec): cosine={cos(full,prefill_decode()):.4f}")

# B) force the dense/prefill memory branch to use the RECURRENT kernel
for l in m.model.layers:
    a=l.attn
    orig=a._memory_branch
    a._memory_branch=types.MethodType(lambda self,*A,**K: type(self)._memory_branch(self,*A,**{**K,"force_recurrent":True}), a)
with torch.no_grad(): full_r=m(ids,use_cache=False).logits[0,-1].float()
print(f"[B] full(REC) vs prefill(REC)+decode(rec):    cosine={cos(full_r,prefill_decode()):.4f}")
print(f"    full(chunk) vs full(REC):                 cosine={cos(full,full_r):.4f}")

# C) direct kernel equivalence on identical delta inputs
torch.manual_seed(1)
B,T,H,D=1,600,16,64
q=torch.randn(B,T,H,D).cuda().bfloat16(); k=torch.randn(B,T,H,D).cuda().bfloat16()
v=torch.randn(B,T,H,D).cuda().bfloat16(); beta=torch.rand(B,T,H).cuda().bfloat16()
g=(-torch.rand(B,T,H).cuda().float()*0.05)
oc,sc=chunk_gated_delta_rule(q=q,k=k,v=v,g=g,beta=beta,output_final_state=True,use_qk_l2norm_in_kernel=True)
orr,sr=fused_recurrent_gated_delta_rule(q=q,k=k,v=v,g=g,beta=beta,output_final_state=True,use_qk_l2norm_in_kernel=True)
print(f"[C] chunk vs recurrent  out cosine={cos(oc[0,-1].reshape(-1),orr[0,-1].reshape(-1)):.4f}  state cosine={cos(sc.reshape(-1),sr.reshape(-1)):.4f}")

# D) state-carry handoff with recurrent kernel (split at P)
P=560
o1,s1=fused_recurrent_gated_delta_rule(q=q[:,:P],k=k[:,:P],v=v[:,:P],g=g[:,:P],beta=beta[:,:P],output_final_state=True,use_qk_l2norm_in_kernel=True)
o2,_=fused_recurrent_gated_delta_rule(q=q[:,P:],k=k[:,P:],v=v[:,P:],g=g[:,P:],beta=beta[:,P:],initial_state=s1,output_final_state=True,use_qk_l2norm_in_kernel=True)
print(f"[D] rec[:P]+rec[P:] vs rec[:] last-out cosine={cos(o2[0,-1].reshape(-1),orr[0,-1].reshape(-1)):.4f}")
