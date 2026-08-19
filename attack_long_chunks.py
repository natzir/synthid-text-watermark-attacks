"""Run 2, long texts: paraphrase / SIRA paragraph by paragraph (whole-text paraphrase with a small model collapses into a summary).
Writes data/attacks_long_chunks.jsonl with paraphrase_ch and sira_ch (chunks re-joined with blank lines). Env: ATTACK_MODEL_OVERRIDE, LIMIT."""
import os, re, sys, time, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import ATTACK_MODEL_TORCH, DATA, device, chat, load_jsonl, dump_jsonl

BS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SIRA_PCT = 30
dev = device()
ATTACKER = os.environ.get("ATTACK_MODEL_OVERRIDE", ATTACK_MODEL_TORCH)
print("attacker:", ATTACKER, flush=True)
tok = AutoTokenizer.from_pretrained(ATTACKER, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(ATTACKER, dtype=torch.float16).to(dev).eval()

def run_batch(users, greedy=False, seed=0, max_new=350):
    outs = []
    for b in range(0, len(users), BS):
        torch.manual_seed(seed + b)
        enc = tok([chat(tok, u) for u in users[b:b + BS]], return_tensors="pt", padding=True).to(dev)
        kw = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
        kw.update(dict(do_sample=False) if greedy else dict(do_sample=True, temperature=0.7, top_p=0.9))
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        outs += [tok.decode(x, skip_special_tokens=True).strip() for x in out[:, enc["input_ids"].shape[1]:]]
    return outs

def p_paraphrase(t):
    return ("You are a paraphraser. You are given an input passage 'INPUT'. You should paraphrase 'INPUT' to print "
            "'OUTPUT'. 'OUTPUT' should be diverse and different as much as possible from 'INPUT' and should not copy "
            "any part verbatim from 'INPUT'. However, 'OUTPUT' should preserve the information in the INPUT. "
            "You should print 'OUTPUT' and nothing else so that it is easy for me to parse.\nINPUT: " + t)
def p_fill(reference, blank):
    return ("You will be shown one reference paragraph and one incomplete paragraph.\n"
            "Your task is to write a complete paragraph using incomplete paragraph.\n"
            "The complete paragraph should have similar length with reference paragraph.\n"
            "You need to include all the information in the reference. \n"
            "But do not take the expression and words in the reference paragraph.\n"
            "You should only answer the complete paragraph.\n"
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
    return tok.decode(ids[0, :1]) + " ".join("_" if s > thr else t for t, s in zip(toks, si))

def chunks(text):
    # paragraphs; headings (short lines) stay attached to the following paragraph
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = []
    for p in parts:
        if out and len(out[-1].split()) < 12:
            out[-1] = out[-1] + "\n\n" + p
        else:
            out.append(p)
    return out

rows = load_jsonl(DATA / "gen_long.jsonl")
if os.environ.get("LIMIT"): rows = rows[:int(os.environ["LIMIT"])]
t0 = time.time()
all_chunks = []   # (row_index, chunk_index, text)
for ri, r in enumerate(rows):
    for ci, c in enumerate(chunks(r["wm"])):
        all_chunks.append((ri, ci, c))
print(f"{len(rows)} texts -> {len(all_chunks)} chunks", flush=True)
paras = run_batch([p_paraphrase(c) for _, _, c in all_chunks], seed=41)
for i, p in enumerate(paras):
    for pref in ("OUTPUT:", "Output:"):
        if p.startswith(pref): p = p[len(pref):].strip()
    paras[i] = p
print(f"paraphrases done {time.time()-t0:.0f}s", flush=True)
blanks = [self_information_mask(c, SIRA_PCT) for _, _, c in all_chunks]
fills = run_batch([p_fill(p, b) for p, b in zip(paras, blanks)], greedy=True, seed=51)
print(f"fills done {time.time()-t0:.0f}s", flush=True)
out_rows = []
for ri, r in enumerate(rows):
    idxs = [i for i, (rj, _, _) in enumerate(all_chunks) if rj == ri]
    out_rows.append({"idx": r["idx"], "prompt": r["prompt"], "wm": r["wm"], "n_chunks": len(idxs),
                     "paraphrase_ch": "\n\n".join(paras[i] for i in idxs),
                     "sira_ch": "\n\n".join(fills[i] for i in idxs)})
dump_jsonl(DATA / "attacks_long_chunks.jsonl", out_rows)
print("done", time.time() - t0)
