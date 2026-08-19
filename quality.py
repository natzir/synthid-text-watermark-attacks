"""Quality of watermarked vs unwatermarked text (paired, same prompt & seed), 7B run:
(A) perplexity under an independent model (Llama-3.1-8B), (B) pairwise LLM-judge preference with position swap,
(C) lexical stats (length, distinct-2/3, repeated 4-grams). Reads data/gen3_<set>.jsonl, writes results/quality.json. Env: LIMIT."""
import json, math, os, sys
import numpy as np
import mlx.core as mx
from mlx_lm import load
from common import load_jsonl, DATA, RESULTS
from mlxwm import ATTACK_MLX, chat_prompt, gen_text

SETS = sys.argv[1:] or ["en", "es"]
model, tok = load(ATTACK_MLX)

def ppl(text):
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < 8: return float("nan")
    logits = model(mx.array(ids)[None])[0]
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = mx.array(ids[1:])
    nll = -mx.take_along_axis(logp[:-1], tgt[:, None], axis=-1)[:, 0]
    return float(math.exp(float(mx.mean(nll.astype(mx.float32)))))

def judge(prompt, a, b, seed):
    q = ("You are an impartial judge. Below is a writing task and two responses, A and B. Decide which response is "
         "better overall (quality of writing, coherence, fluency, how well it fulfils the task). Answer with exactly one "
         "letter: A or B.\n\nTASK:\n" + prompt + "\n\nRESPONSE A:\n" + a + "\n\nRESPONSE B:\n" + b + "\n\nBetter response (A or B):")
    out = gen_text(model, tok, chat_prompt(tok, q), 3, False, seed=seed, greedy=True).strip().upper()
    return "A" if out.startswith("A") else ("B" if out.startswith("B") else "?")

def lex(text):
    w = text.split(); n = len(w)
    d2 = len(set(zip(w, w[1:]))) / max(1, n - 1); d3 = len(set(zip(w, w[1:], w[2:]))) / max(1, n - 2)
    g4 = list(zip(w, w[1:], w[2:], w[3:])); rep4 = 1 - len(set(g4)) / max(1, len(g4))
    return n, d2, d3, rep4

res = {}
for S in SETS:
    rows = load_jsonl(DATA / f"gen3_{S}.jsonl")
    if os.environ.get("LIMIT"): rows = rows[:int(os.environ["LIMIT"])]  # smoke test
    ppl_wm, ppl_no, lex_wm, lex_no = [], [], [], []
    wins_wm = wins_no = ties = 0; consistent = 0
    for r in rows:
        ppl_wm.append(ppl(r["wm"])); ppl_no.append(ppl(r["nowm"]))
        lex_wm.append(lex(r["wm"])); lex_no.append(lex(r["nowm"]))
        # judge twice with swapped positions; a "win" needs both orders to agree
        j1 = judge(r["prompt"], r["wm"], r["nowm"], seed=1)   # A=wm
        j2 = judge(r["prompt"], r["nowm"], r["wm"], seed=2)   # A=nowm
        pref1 = "wm" if j1 == "A" else ("no" if j1 == "B" else "?")
        pref2 = "no" if j2 == "A" else ("wm" if j2 == "B" else "?")
        if pref1 == pref2 == "wm": wins_wm += 1; consistent += 1
        elif pref1 == pref2 == "no": wins_no += 1; consistent += 1
        else: ties += 1
        print(f"{S} idx={r['idx']} ppl wm={ppl_wm[-1]:.1f} no={ppl_no[-1]:.1f} judge={pref1}/{pref2}", flush=True)
    a, b = np.array(ppl_wm), np.array(ppl_no)
    d = a - b
    # paired sign test / Wilcoxon
    try:
        from scipy.stats import wilcoxon
        pval = float(wilcoxon(a, b).pvalue)
    except Exception:
        pval = float("nan")
    L1, L0 = np.array(lex_wm), np.array(lex_no)
    res[S] = {
        "n": len(rows),
        "ppl_wm_mean": float(np.nanmean(a)), "ppl_no_mean": float(np.nanmean(b)),
        "ppl_wm_median": float(np.nanmedian(a)), "ppl_no_median": float(np.nanmedian(b)),
        "ppl_diff_mean": float(np.nanmean(d)), "ppl_wm_higher_frac": float(np.mean(a > b)), "ppl_wilcoxon_p": pval,
        "judge_wm_wins": wins_wm, "judge_nowm_wins": wins_no, "judge_ties_or_inconsistent": ties,
        "len_wm": float(L1[:, 0].mean()), "len_no": float(L0[:, 0].mean()),
        "distinct2_wm": float(L1[:, 1].mean()), "distinct2_no": float(L0[:, 1].mean()),
        "distinct3_wm": float(L1[:, 2].mean()), "distinct3_no": float(L0[:, 2].mean()),
        "rep4_wm": float(L1[:, 3].mean()), "rep4_no": float(L0[:, 3].mean()),
    }
    print(json.dumps(res[S], indent=1))
json.dump(res, open(RESULTS / "quality.json", "w"), indent=1)
