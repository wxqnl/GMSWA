import json, torch, fla
from fla.models.gated_mem_swa import GatedMemSWAConfig, GatedMemSWAForCausalLM
from fla.models.utils import Cache
def cos(a,b): return torch.nn.functional.cosine_similarity(a.float(),b.float(),dim=0).item()

d=json.load(open("flash-linear-attention/flame/configs/gated_mem_swa_v3_340M.json"))
d["disable_memory"]=True; d["num_hidden_layers"]=2     # tiny: isolate SWA decode
torch.manual_seed(0); cfg=GatedMemSWAConfig(**d)
m=GatedMemSWAForCausalLM(cfg).cuda().to(torch.bfloat16).eval()

# capture o_local at last position in layer 0 for whichever path runs
cap={}
a0=m.model.layers[0].attn
orig=a0._local_attention
def wrap(self,*A,**K):
    o=orig(*A,**K); cap['o']=o[0,-1].reshape(-1).detach().float(); cap['Tkv']=K.get('k_rope',A[1] if len(A)>1 else None)
    return o
import types
a0._local_attention=types.MethodType(lambda self,**K: (_ for _ in ()).throw(Exception()) , a0)  # placeholder
a0._local_attention=types.MethodType(lambda self,*A,**K: wrap(self,*A,**K), a0)

for N in [520, 600]:
    ids=torch.randint(0,cfg.vocab_size,(1,N)).cuda()
    with torch.no_grad():
        full=m(ids,use_cache=False).logits[0,-1].float(); o_full=cap['o'].clone(); tkv_full=cap['Tkv'].shape[1]
        pkv=Cache.from_legacy_cache(None)
        m(ids[:,:N-1],use_cache=True,past_key_values=pkv); o_pre=cap['o'].clone(); tkv_pre=cap['Tkv'].shape[1]
        dec=m(ids[:,N-1:N],use_cache=True,past_key_values=pkv).logits[0,-1].float(); o_dec=cap['o'].clone(); tkv_dec=cap['Tkv'].shape[1]
    print(f"N={N}: logits cos(full,dec)={cos(full,dec):.4f} | o_local cos(full,dec)={cos(o_full,o_dec):.4f}")
    print(f"      T_kv seen: full={tkv_full} prefill={tkv_pre} decode={tkv_dec}  (window={cfg.window_size})")
