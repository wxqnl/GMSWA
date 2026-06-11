"""Isolate the prefill-vs-decode mismatch: baseline (shared proj) vs v3 (separate),
at N<window (no eviction) and N>window (eviction). Same harness for all."""
import json, torch, fla  # noqa
from fla.models.gated_mem_swa import GatedMemSWAConfig, GatedMemSWAForCausalLM
from fla.models.utils import Cache

base=json.load(open("flash-linear-attention/flame/configs/gated_mem_swa_340M.json"))   # shared proj, old gates
v3=json.load(open("flash-linear-attention/flame/configs/gated_mem_swa_v3_340M.json"))   # separate proj

def consistency(cfg_dict, N):
    torch.manual_seed(0)
    cfg=GatedMemSWAConfig(**cfg_dict)
    m=GatedMemSWAForCausalLM(cfg).cuda().to(torch.bfloat16).eval()
    ids=torch.randint(0,cfg.vocab_size,(1,N)).cuda()
    with torch.no_grad():
        full=m(ids,use_cache=False).logits[0,-1].float()
        pkv=Cache.from_legacy_cache(None)
        m(ids[:,:N-1],use_cache=True,past_key_values=pkv)
        dec=m(ids[:,N-1:N],use_cache=True,past_key_values=pkv).logits[0,-1].float()
    cos=torch.nn.functional.cosine_similarity(full,dec,dim=0).item()
    del m; torch.cuda.empty_cache()
    return cos,(full-dec).abs().max().item()

for name,cfg in [("baseline(shared)",base),("v3(separate)",v3)]:
    for N in [400, 600]:   # 400 < window(512) => no eviction; 600 > window => eviction
        evict = "no-evict" if N<512 else "EVICT"
        c,d=consistency(cfg,N)
        print(f"{name:18s} N={N} ({evict:8s}): cosine={c:.5f}  max|d|={d:.4f}")
