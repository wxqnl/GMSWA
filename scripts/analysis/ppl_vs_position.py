"""Long-context utilization for PT base models (no SFT needed).
Per-position NLL over long documents: a pure-SWA model's loss should PLATEAU
beyond the 512 window; a model whose memory uses far context keeps improving.
Compares SWA / GMSWA-v2 / GMSWA-v3 on the SAME documents, clean forward path.

Usage: python ppl_vs_position.py <model_name> <ckpt_dir>   (writes a json line)
"""
import sys, json, torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

NAME, CKPT = sys.argv[1], sys.argv[2]
SEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
NDOC = int(sys.argv[4]) if len(sys.argv) > 4 else 120
BUCKETS = [(0,512),(512,1024),(1024,2048),(2048,4096),(4096,8192)]

tok = AutoTokenizer.from_pretrained(CKPT)
# collect NDOC documents with >= SEQ tokens from SlimPajama (deterministic order)
import glob
_shards = sorted(glob.glob("/home/user01/Minko/datasets/SlimPajama-627B/data/train-*.parquet"))[:16]
ds = load_dataset("parquet", data_files=_shards, split="train", streaming=True)
LONG_SRC = {"RedPajamaBook", "RedPajamaArXiv"}   # sources that are reliably long
docs = []
for ex in ds:
    if ex.get("meta", {}).get("redpajama_set_name") not in LONG_SRC:
        continue   # skip short web docs cheaply (no tokenization)
    ids = tok(ex["text"], return_tensors="pt").input_ids[0]
    if ids.numel() >= SEQ:
        docs.append(ids[:SEQ])
    if len(docs) >= NDOC:
        break
print(f"[{NAME}] collected {len(docs)} docs of {SEQ} tokens", flush=True)

model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
sums = [0.0]*len(BUCKETS); cnts = [0]*len(BUCKETS)
import torch.nn.functional as F
for d in docs:
    ids = d.unsqueeze(0).cuda()
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits[0].float()
    nll = F.cross_entropy(logits[:-1], ids[0,1:], reduction="none")  # per-token, target pos = 1..SEQ-1
    pos = torch.arange(1, ids.shape[1], device=nll.device)
    for bi,(lo,hi) in enumerate(BUCKETS):
        m = (pos>=lo)&(pos<hi)
        sums[bi]+=nll[m].sum().item(); cnts[bi]+=int(m.sum())
res = {NAME: {f"{lo}-{hi}": (sums[bi]/cnts[bi] if cnts[bi] else None) for bi,(lo,hi) in enumerate(BUCKETS)}}
print("RESULT " + json.dumps(res), flush=True)
