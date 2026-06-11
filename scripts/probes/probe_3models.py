"""Immediate signal: does v3's memory recall a needle placed OUTSIDE the 512
window, where SWA structurally can't and v2's memory failed? Uses the now-fixed
generation path. SWA = no-memory baseline; v2 = broken memory; v3 = the fix."""
import torch, fla  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer
B="flash-linear-attention/flame/saves"
MODELS=[("SWA", f"{B}/SWA-340M-v2-10k"), ("GMSWA-v2", f"{B}/GMSWA-340M-v2-10k"), ("GMSWA-v3", f"{B}/GMSWA-340M-v3-10k")]
NEEDLE="8137"
filler="The garden was quiet and the wind moved slowly through the old trees. "
def build(tok, dist):
    pre=filler*8+f"Important: the secret access token is {NEEDLE}. "+filler*max(1,dist//14)
    return tok(pre+"Question: the secret access token is",return_tensors="pt").input_ids.cuda()
print(f"needle={NEEDLE}; '.'=miss 'Y'=recalled")
print(f"{'model':10s} | {'d~200(IN)':>10s} {'d~800(OUT)':>10s} {'d~1500(OUT)':>11s}")
for name,ck in MODELS:
    tok=AutoTokenizer.from_pretrained(ck)
    m=AutoModelForCausalLM.from_pretrained(ck,dtype=torch.bfloat16,trust_remote_code=True).cuda().eval()
    row=[]
    for dist in [200,800,1500]:
        ids=build(tok,dist)
        with torch.no_grad():
            out=m.generate(ids,max_new_tokens=6,do_sample=False,pad_token_id=tok.eos_token_id,use_cache=True)
        g=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True)
        row.append("Y" if NEEDLE in g else f".[{g.strip()[:6]}]")
    print(f"{name:10s} | {row[0]:>10s} {row[1]:>10s} {row[2]:>11s}")
    del m; torch.cuda.empty_cache()
