"""Master analysis: reads eval_results/suite/<model>/ and ppl_*.out, prints the
paper's core comparison tables across all models."""
import json, glob, sys
MODELS = sys.argv[1:] or ["SWA","Transformer","GMSWA-v2","GMSWA-v3","GDN"]
LENS=[512,1024,2048,4096,8192]
SING=["niah_single_1","niah_single_2","niah_single_3"]
def niah(m):
    o={}
    for L in LENS:
        fs=sorted(glob.glob(f"eval_results/suite/{m}/ruler_{L}/**/results*.json",recursive=True))
        if not fs: continue
        r=json.load(open(fs[-1]))["results"]
        v=[r[t][f"{L},none"] for t in SING if t in r and isinstance(r[t].get(f"{L},none"),(int,float)) and r[t][f"{L},none"]>=0]
        if v: o[L]=sum(v)/len(v)
    return o
def recall(m):
    fs=sorted(glob.glob(f"eval_results/suite/{m}/recall/**/results*.json",recursive=True))
    if not fs: return {}
    r=json.load(open(fs[-1]))["results"]; o={}
    for t in ["swde","fda","squad_completion"]:
        if t in r:
            for k,val in r[t].items():
                if isinstance(val,(int,float)) and not k.endswith("_stderr"): o[t]=val; break
    return o
print("\n================ NIAH single-needle recall vs length ================")
print(f"{'model':12s} "+" ".join(f"{L:>6}" for L in LENS))
data={m:niah(m) for m in MODELS}
for m in MODELS:
    if data[m]: print(f"{m:12s} "+" ".join(f"{data[m].get(L,float('nan')):6.2f}" for L in LENS))
print("\n================ recall-intensive real tasks ================")
print(f"{'model':12s} {'swde':>7} {'fda':>7} {'squad':>7}")
for m in MODELS:
    r=recall(m)
    if r: print(f"{m:12s} {r.get('swde',float('nan')):7.3f} {r.get('fda',float('nan')):7.3f} {r.get('squad_completion',float('nan')):7.3f}")
print("\n================ loss-vs-position (from ppl_*.out if present) ================")
import re
PB=[(0,512),(512,1024),(1024,2048),(2048,4096),(4096,8192)]
for m in MODELS:
    # exact filename only — substring glob over-matches (e.g. 'SWA' hits 'GMSWA-v2/v3')
    for f in sorted(set(glob.glob(f"eval_results/ppl_{m}.out"))):
        for line in open(f):
            if line.startswith("RESULT"):
                d=json.loads(line[7:]); k=list(d)[0]
                print(f"{m:12s} "+" ".join(f"{d[k][f'{a}-{b}']:.3f}" for a,b in PB if f'{a}-{b}' in d[k]))
                break

# ---- standard zero-shot LM benchmark ("short" suite) ----
SHORTDIR={"SWA":"SWA-340M-v2-10k","Transformer":"Transformer-340M-10k",
          "GMSWA-v2":"GMSWA-340M-v2-10k","GMSWA-v3":"GMSWA-340M-v3-10k",
          "GDN":"GDN-340M-10k","GMSWA-v3-1B":"GMSWA-v3-1B-10k"}
STASKS=["arc_challenge","arc_easy","boolq","copa","hellaswag","lambada_openai",
        "openbookqa","piqa","sciq","winogrande"]
def short_bench(m):
    d=SHORTDIR.get(m)
    if not d: return {}
    fs=sorted(glob.glob(f"eval_results/{d}/short/**/results*.json",recursive=True))
    if not fs: return {}
    r=json.load(open(fs[-1]))["results"]; o={}
    for t in STASKS:
        if t in r:
            v=r[t].get("acc,none", r[t].get("acc_norm,none"))
            if isinstance(v,(int,float)): o[t]=v
    if "wikitext" in r:
        o["wikitext_ppl"]=r["wikitext"].get("word_perplexity,none", r["wikitext"].get("perplexity,none"))
    return o
print("\n================ standard zero-shot benchmark (acc; wikitext_ppl unreliable) ================")
print(f"{'model':12s} "+" ".join(f"{t[:5]:>6}" for t in STASKS)+f" {'avg':>6} {'wiki_ppl':>9}")
for m in MODELS:
    s=short_bench(m)
    if not s: continue
    accs=[s[t] for t in STASKS if t in s]
    avg=sum(accs)/len(accs) if accs else float('nan')
    print(f"{m:12s} "+" ".join(f"{s.get(t,float('nan')):6.3f}" for t in STASKS)
          +f" {avg:6.3f} {s.get('wikitext_ppl',float('nan')):9.1f}")
