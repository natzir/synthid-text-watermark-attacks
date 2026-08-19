"""Run 3 (MLX): attacks with Llama-3.1-8B-Instruct-4bit. Usage: python3 attack3.py <set> [limit] -> data/attacks3_<set>.jsonl (resumable)
en: paraphrase, SIRA (all) + round-trip ES/ZH (first N_RT); es: paraphrase, SIRA, round-trip EN/ZH; long: paragraph-wise paraphrase + SIRA."""
import re, sys, time, numpy as np
import mlx.core as mx
from mlx_lm import load
from common import dump_jsonl, load_jsonl, DATA
from prompts2 import SETS
from mlxwm import ATTACK_MLX, chat_prompt, gen_text

SET = sys.argv[1]; LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
N_RT = 50; SIRA_PCT = 30
cfg = SETS[SET]; LANG = cfg["lang"]
MAX_NEW = 420
model, tok = load(ATTACK_MLX)

def run(user, greedy=False, seed=0, max_new=MAX_NEW):
    return gen_text(model, tok, chat_prompt(tok, user), max_new, False, seed=seed, temp=0.7, top_k=0, greedy=greedy).strip()

LANG_NOTE = " Write the OUTPUT in Spanish, the same language as the INPUT." if LANG == "es" else ""
def p_paraphrase(t):
    n = len(t.split())
    return ("You are a paraphraser. You are given an input passage 'INPUT'. You should paraphrase 'INPUT' to print "
            "'OUTPUT'. 'OUTPUT' should be diverse and different as much as possible from 'INPUT' and should not copy "
            "any part verbatim from 'INPUT'. However, 'OUTPUT' should preserve the information in the INPUT. "
            f"'OUTPUT' must be a full paraphrase, not a summary: keep every point and about the same length (around {n} words). "
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

def clean(t):
    t = t.strip()
    for pref in ("OUTPUT:", "Output:", "'OUTPUT':", "OUTPUT"):
        if t.startswith(pref): t = t[len(pref):].strip()
    return t

def self_information_mask(text, pct):
    ids = tok.encode(text, add_special_tokens=False)
    logits = model(mx.array(ids)[None])[0]                    # (n, V)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = mx.array(ids[1:])
    si = -mx.take_along_axis(logp[:-1], tgt[:, None], axis=-1)[:, 0]
    si = np.array(si.astype(mx.float32)).tolist()
    toks = [tok.decode([i]) for i in ids[1:]]
    thr = np.percentile(si, pct)
    return tok.decode(ids[:1]) + " ".join("_" if s > thr else t for t, s in zip(toks, si))

def chunks(text):
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = []
    for p in parts:
        if out and len(out[-1].split()) < 12: out[-1] = out[-1] + "\n\n" + p
        else: out.append(p)
    return out

src_lang = "Spanish" if LANG == "es" else "English"
pivots = {"en": [("es", "Spanish"), ("zh", "Simplified Chinese")], "es": [("en", "English"), ("zh", "Simplified Chinese")]}.get(SET, [])

rows = load_jsonl(DATA / f"gen3_{SET}.jsonl")
if LIMIT: rows = rows[:LIMIT]
try: done = {a["idx"]: a for a in load_jsonl(DATA / f"attacks3_{SET}.jsonl")}
except FileNotFoundError: done = {}
out = list(done.values()); t0 = time.time()
for r in rows:
    if r["idx"] in done: continue
    wm = r["wm"]; a = {"idx": r["idx"], "prompt": r["prompt"], "wm": wm}
    if SET == "long":
        cs = chunks(wm)
        paras = [clean(run(p_paraphrase(c), seed=100 + j)) for j, c in enumerate(cs)]
        fills = [clean(run(p_fill(p, self_information_mask(c, SIRA_PCT)), greedy=True)) for p, c in zip(paras, cs)]
        a["n_chunks"] = len(cs); a["paraphrase_ch"] = "\n\n".join(paras); a["sira_ch"] = "\n\n".join(fills)
    else:
        a["paraphrase"] = clean(run(p_paraphrase(wm), seed=1))
        if SET != "en" or r["idx"] < N_RT:
            for code, name in pivots:
                mid = run(p_translate(wm, name), seed=2, max_new=520)
                a[f"rt_{code}_mid"] = mid; a[f"rt_{code}"] = clean(run(p_translate(mid, src_lang), seed=3, max_new=520))
        a["sira_blank"] = self_information_mask(wm, SIRA_PCT)
        a["sira"] = clean(run(p_fill(a["paraphrase"], a["sira_blank"]), greedy=True))
    out.append(a); dump_jsonl(DATA / f"attacks3_{SET}.jsonl", out)
    print(f"[{len(out)}/{len(rows)}] {SET} idx={r['idx']}  {time.time()-t0:.0f}s", flush=True)
print("done", SET, time.time() - t0)
