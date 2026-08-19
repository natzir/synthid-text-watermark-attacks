"""Inter-response diversity: Self-BLEU (each response vs the other k-1), mean pairwise 4-gram Jaccard and
mean pairwise embedding cosine, watermarked vs unwatermarked, per temperature. Reads data/diversity.jsonl -> results/diversity.json"""
import json, itertools
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from common import load_jsonl, DATA, RESULTS

rows = load_jsonl(DATA / "diversity.jsonl")
st = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
sm = SmoothingFunction().method1
def toks(t): return t.lower().split()
def self_bleu(resps):
    tk = [toks(r) for r in resps]; out = []
    for i in range(len(tk)):
        refs = [tk[j] for j in range(len(tk)) if j != i]
        out.append(sentence_bleu(refs, tk[i], smoothing_function=sm))
    return float(np.mean(out)) * 100
def jacc4(resps):
    sets = [set(zip(*[toks(r)[i:] for i in range(4)])) for r in resps]; out = []
    for a, b in itertools.combinations(sets, 2):
        out.append(len(a & b) / max(1, len(a | b)))
    return float(np.mean(out)) * 100
def emb_cos(resps):
    e = st.encode(resps, normalize_embeddings=True); out = []
    for a, b in itertools.combinations(range(len(resps)), 2):
        out.append(float(np.dot(e[a], e[b])))
    return float(np.mean(out))

keys = [k for k in rows[0] if k.startswith(("wm_", "nowm_"))]
res = {}
for k in keys:
    sb = [self_bleu(r[k]) for r in rows]; jc = [jacc4(r[k]) for r in rows]; ec = [emb_cos(r[k]) for r in rows]
    res[k] = {"n_prompts": len(rows), "self_bleu": float(np.mean(sb)), "jaccard4_pct": float(np.mean(jc)), "emb_cos": float(np.mean(ec)),
              "per_prompt_self_bleu": sb}
    print(f"{k:12s} Self-BLEU={res[k]['self_bleu']:5.1f}  4-gram Jaccard={res[k]['jaccard4_pct']:5.1f}%  emb-cos={res[k]['emb_cos']:.3f}")
# paired comparison wm vs nowm per temperature
for T in ("T1.0", "T0.7"):
    a = np.array(res[f"wm_{T}"]["per_prompt_self_bleu"]); b = np.array(res[f"nowm_{T}"]["per_prompt_self_bleu"])
    try:
        from scipy.stats import wilcoxon; p = float(wilcoxon(a, b).pvalue)
    except Exception: p = float("nan")
    print(f"{T}: Self-BLEU wm-nowm mean diff = {float(np.mean(a-b)):+.1f}; wm higher in {float(np.mean(a>b)):.0%} of prompts; Wilcoxon p={p:.3g}")
    res[f"paired_{T}"] = {"diff_mean": float(np.mean(a - b)), "wm_higher_frac": float(np.mean(a > b)), "wilcoxon_p": p}
json.dump(res, open(RESULTS / "diversity.json", "w"), indent=1)
