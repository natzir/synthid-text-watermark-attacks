"""Score every variant (sets en / es / long / low) with the SynthID mean-score detector and summarise.
RUN="" (default) reads data/gen_*.jsonl + data/attacks_*.jsonl (run 2); RUN=3 reads data/gen3_*/attacks3_* (run 3).
Writes results/results{2|3}.json and results/results{2|3}_table.md. Charts: report2.py."""
import json, math, random, os
RUN = os.environ.get("RUN", "")  # "" -> gen_/attacks_ files; "3" -> gen3_/attacks3_ files
import numpy as np
from transformers import AutoTokenizer
from common import TOKENIZER_MODEL, NGRAM_LEN, DATA, RESULTS, load_jsonl, unicode_clean
from score import score

Z_1PCT, Z_KGW = 2.326, 4.0
SETS = ["en", "es", "long", "low"]
LABELS = {
    "nowm": "Sin marca (control)", "wm": "Con marca, sin tocar", "wm_unicode_cleaner": "Con marca + limpiador Unicode",
    "wm_word_delete30": "Borrar 30 % de palabras", "paraphrase": "Paráfrasis con otro modelo",
    "rt_es": "Traducción ida-vuelta por español", "rt_en": "Traducción ida-vuelta por inglés",
    "rt_zh": "Traducción ida-vuelta por chino", "sira": "SIRA (máscara 70 % + relleno)",
    "paraphrase_ch": "Paráfrasis párrafo a párrafo", "sira_ch": "SIRA párrafo a párrafo",
}
ORDER = ["nowm", "wm", "wm_unicode_cleaner", "wm_word_delete30", "rt_es", "rt_en", "rt_zh", "paraphrase", "sira", "paraphrase_ch", "sira_ch"]

tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
def ngrams(t, n=NGRAM_LEN):
    ids = tok(t, add_special_tokens=False)["input_ids"]
    return set(tuple(ids[i:i + n]) for i in range(len(ids) - n + 1))
def delete_words(t, frac, seed):
    rnd = random.Random(seed)
    return " ".join(x for x in t.split() if rnd.random() > frac)

from sentence_transformers import SentenceTransformer
st = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
def sim(a, b):
    e = st.encode([a, b], normalize_embeddings=True)
    return float(np.dot(e[0], e[1]))

results, items = {}, []
for S in SETS:
    try:
        gen = load_jsonl(DATA / f"gen{RUN}_{S}.jsonl")
    except FileNotFoundError:
        continue
    try:
        att = {a["idx"]: a for a in load_jsonl(DATA / f"attacks{RUN}_{S}.jsonl")}
    except FileNotFoundError:
        att = {}
    if S == "long" and not RUN:
        try:
            for a in load_jsonl(DATA / "attacks_long_chunks.jsonl"):
                att.setdefault(a["idx"], {}).update({k: a[k] for k in ("paraphrase_ch", "sira_ch")})
        except FileNotFoundError:
            pass
    variants = {}
    for r in gen:
        wm = r["wm"]
        variants.setdefault("wm", []).append((r["idx"], wm, wm))
        variants.setdefault("nowm", []).append((r["idx"], r["nowm"], wm))
        variants.setdefault("wm_unicode_cleaner", []).append((r["idx"], unicode_clean(wm), wm))
        variants.setdefault("wm_word_delete30", []).append((r["idx"], delete_words(wm, 0.3, r["idx"]), wm))
        if r["idx"] in att:
            a = att[r["idx"]]
            for k in ["paraphrase", "rt_es", "rt_en", "rt_zh", "sira", "paraphrase_ch", "sira_ch"]:
                if k in a and a[k].strip():
                    t = a[k].strip()
                    for pref in ("OUTPUT:", "Output:", "'OUTPUT':", "OUTPUT"):
                        if t.startswith(pref):
                            t = t[len(pref):].strip()
                    variants.setdefault(k, []).append((r["idx"], t, wm))
    for name, lst in variants.items():
        zs, mgs, sims, ntok, ovs = [], [], [], [], []
        for idx, text, orig in lst:
            s = score(text)
            if math.isnan(s["z"]):
                continue
            base = ngrams(orig); g = ngrams(text)
            ov = len(g & base) / max(1, len(g))
            sm = 1.0 if text == orig else (sim(text, orig) if name != "nowm" else float("nan"))
            zs.append(s["z"]); mgs.append(s["mean_g"]); ntok.append(s["n_tokens"]); ovs.append(ov); sims.append(sm)
            items.append({"set": S, "variant": name, "idx": idx, "z": s["z"], "mean_g": s["mean_g"],
                          "n_tokens": s["n_tokens"], "overlap5": ov, "sim": sm})
        zs = np.array(zs)
        results[f"{S}:{name}"] = {
            "set": S, "variant": name, "n": int(len(zs)), "tokens_mean": float(np.mean(ntok)),
            "mean_g": float(np.mean(mgs)), "z_mean": float(zs.mean()), "z_median": float(np.median(zs)),
            "det_rate_1pct": float((zs >= Z_1PCT).mean()), "det_rate_z4": float((zs >= Z_KGW).mean()),
            "sim_mean": float(np.nanmean(sims)) if name != "nowm" else float("nan"),
            "overlap5_mean": float(np.mean(ovs)),
        }

json.dump({"summary": results, "items": items}, open(RESULTS / f"results{RUN or 2}.json", "w"), indent=1)

lines = []
for S in SETS:
    keys = [f"{S}:{v}" for v in ORDER if f"{S}:{v}" in results]
    if not keys: continue
    lines.append(f"\n### Set `{S}`\n")
    lines.append("| Variante | n | tokens | media g | z mediana | detectado (z≥2,33) | detectado (z≥4) | similitud | 5-gramas que sobreviven |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in keys:
        r = results[k]
        simtxt = "—" if math.isnan(r["sim_mean"]) else f"{r['sim_mean']:.2f}"
        ovtxt = "—" if r["variant"] == "nowm" else f"{r['overlap5_mean']:.0%}"
        lines.append(f"| {LABELS[r['variant']]} | {r['n']} | {r['tokens_mean']:.0f} | {r['mean_g']:.3f} | {r['z_median']:.1f} | "
                     f"{r['det_rate_1pct']:.0%} | {r['det_rate_z4']:.0%} | {simtxt} | {ovtxt} |")
    print("\n".join(lines[-(len(keys) + 3):]))
open(RESULTS / f"results{RUN or 2}_table.md", "w").write("\n".join(lines) + "\n")

# correlation overlap vs z over attacked items (excluding wm/nowm/unicode)
att_items = [i for i in items if i["variant"] not in ("wm", "nowm", "wm_unicode_cleaner")]
if att_items:
    a = np.array([(i["overlap5"], i["z"]) for i in att_items])
    print("\nPearson r(5-gram overlap, z) over", len(a), "attacked texts:", round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3))
