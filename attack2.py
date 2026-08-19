"""Run 2 (PyTorch/MPS): batched black-box attacks with an open model that does not know the key. Usage: python3 attack2.py <set> [batch_size]
Attacks per set: en -> paraphrase, SIRA (all) + round-trip ES/ZH (first N_RT rows); es -> paraphrase, SIRA, round-trip EN/ZH;
long -> paraphrase, SIRA. Writes data/attacks_<set>.jsonl (resumable). Env: ATTACK_MODEL_OVERRIDE (HF id), LIMIT=<n> (smoke test)."""
import os, sys, time, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import ATTACK_MODEL_TORCH, DATA, device, chat, load_jsonl, dump_jsonl
from prompts2 import SETS

SET = sys.argv[1]
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
N_RT = 50           # rows that also get the translation attacks (en set only)
SIRA_PCT = 30
cfg = SETS[SET]
LANG = cfg["lang"]
MAX_NEW = 1300 if SET == "long" else 400
dev = device()
ATTACKER = os.environ.get("ATTACK_MODEL_OVERRIDE", ATTACK_MODEL_TORCH)
print("attacker:", ATTACKER, flush=True)
tok = AutoTokenizer.from_pretrained(ATTACKER, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(ATTACKER, dtype=torch.float16).to(dev).eval()

def run_batch(users, greedy=False, seed=0):
    torch.manual_seed(seed)
    enc = tok([chat(tok, u) for u in users], return_tensors="pt", padding=True).to(dev)
    kw = dict(max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
    kw.update(dict(do_sample=False) if greedy else dict(do_sample=True, temperature=0.7, top_p=0.9))
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    return [tok.decode(x, skip_special_tokens=True).strip() for x in out[:, enc["input_ids"].shape[1]:]]

LANG_NOTE = " Write the OUTPUT in Spanish, the same language as the INPUT." if LANG == "es" else ""
def p_paraphrase(t):
    return ("You are a paraphraser. You are given an input passage 'INPUT'. You should paraphrase 'INPUT' to print "
            "'OUTPUT'. 'OUTPUT' should be diverse and different as much as possible from 'INPUT' and should not copy "
            "any part verbatim from 'INPUT'. However, 'OUTPUT' should preserve the information in the INPUT. "
            "You should print 'OUTPUT' and nothing else so that it is easy for me to parse." + LANG_NOTE + "\nINPUT: " + t)
def p_translate(t, lang):
    return (f"Translate the following text into {lang}. Preserve the meaning and structure. "
            f"Output only the translation, nothing else.\n\nTEXT:\n{t}")
def p_fill(reference, blank):
    return ("You will be shown one reference paragraph and one incomplete paragraph.\n"
            "Your task is to write a complete paragraph using incomplete paragraph.\n"
            "The complete paragraph should have similar length with reference paragraph.\n"
            "You need to include all the information in the reference. \n"
            "But do not take the expression and words in the reference paragraph.\n"
            "You should only answer the complete paragraph." + (" Write in Spanish." if LANG == "es" else "") + "\n"
            f"reference: {reference}\n"
            f"incomplete pragraph: {blank}\n")

def self_information_mask(text, pct):
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
    with torch.no_grad():
        logits = model(**enc).logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    ids = enc["input_ids"]
    si = (-logp[0, :-1].gather(-1, ids[0, 1:, None]).squeeze(-1)).tolist()
    toks = [tok.decode([i]) for i in ids[0, 1:].tolist()]
    thr = np.percentile(si, pct)
    out = ["_" if s > thr else t for t, s in zip(toks, si)]
    return tok.decode(ids[0, :1]) + " ".join(out)

src_lang_name = "Spanish" if LANG == "es" else "English"
pivots = {"en": [("es", "Spanish"), ("zh", "Simplified Chinese")], "es": [("en", "English"), ("zh", "Simplified Chinese")], "long": []}[SET if SET in ("en", "es", "long") else "long"]

rows = load_jsonl(DATA / f"gen_{SET}.jsonl")
if os.environ.get("LIMIT"): rows = rows[:int(os.environ["LIMIT"])]
try:
    done = {a["idx"]: a for a in load_jsonl(DATA / f"attacks_{SET}.jsonl")}
except FileNotFoundError:
    done = {}
out_rows = list(done.values())
todo = [r for r in rows if r["idx"] not in done]
t0 = time.time()
for b in range(0, len(todo), BS):
    chunk = todo[b:b + BS]
    wms = [r["wm"] for r in chunk]
    res = [{"idx": r["idx"], "prompt": r["prompt"], "wm": r["wm"]} for r in chunk]
    paras = run_batch([p_paraphrase(t) for t in wms], seed=11 + b)
    for a, p in zip(res, paras): a["paraphrase"] = p
    for code, name in pivots:
        sel = [i for i, r in enumerate(chunk) if SET != "en" or r["idx"] < N_RT]
        if not sel: continue
        mid = run_batch([p_translate(wms[i], name) for i in sel], seed=21 + b)
        back = run_batch([p_translate(m, src_lang_name) for m in mid], seed=31 + b)
        for i, m, bk in zip(sel, mid, back):
            res[i][f"rt_{code}_mid"] = m; res[i][f"rt_{code}"] = bk
    blanks = [self_information_mask(t, SIRA_PCT) for t in wms]
    fills = run_batch([p_fill(a["paraphrase"], bl) for a, bl in zip(res, blanks)], greedy=True)
    for a, bl, f in zip(res, blanks, fills): a["sira_blank"] = bl; a["sira"] = f
    out_rows.extend(res)
    dump_jsonl(DATA / f"attacks_{SET}.jsonl", out_rows)
    print(f"[{len(out_rows)}/{len(rows)}] {SET} batch done  {time.time()-t0:.0f}s", flush=True)
print("done", SET, time.time() - t0)
