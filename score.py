"""SynthID-Text detection: mean g-value score (Nature paper's simple 'mean' detector) with repeated-context masking.
Under no watermark each g-value ~ Bernoulli(0.5), so z = (mean-0.5)/(0.5/sqrt(N)) with N = unmasked positions * depth."""
import math
from transformers import AutoTokenizer
from common import TOKENIZER_MODEL, NGRAM_LEN, DATA, wm_processor, load_jsonl

_tok = None
_proc = None

def _init():
    global _tok, _proc
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        _proc = wm_processor("cpu")

def score(text):
    """Returns dict(mean_g, z, n_tokens, n_scored)."""
    _init()
    ids = _tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    n = ids.shape[1]
    if n < NGRAM_LEN + 1:
        return {"mean_g": float("nan"), "z": float("nan"), "n_tokens": n, "n_scored": 0}
    g = _proc.compute_g_values(ids)                       # (1, n-(ngram-1), depth)
    mask = _proc.compute_context_repetition_mask(ids)     # (1, n-(ngram-1))
    depth = g.shape[-1]
    m = mask.float()
    n_scored = int(m.sum().item())
    if n_scored == 0:
        return {"mean_g": float("nan"), "z": float("nan"), "n_tokens": n, "n_scored": 0}
    mean_g = float((g.float() * m[:, :, None]).sum() / (depth * n_scored))
    z = (mean_g - 0.5) / (0.5 / math.sqrt(n_scored * depth))
    return {"mean_g": mean_g, "z": z, "n_tokens": n, "n_scored": n_scored}

if __name__ == "__main__":
    # usage: python3 score.py data/gen3_en.jsonl   -> z-score of every text field in each row
    import sys
    rows = load_jsonl(sys.argv[1] if len(sys.argv) > 1 else DATA / "gen3_en.jsonl")
    for r in rows:
        for k, v in r.items():
            if isinstance(v, str) and k not in ("set", "prompt") and len(v) > 40:
                s = score(v)
                print(f"{r.get('idx', '?'):>3} {k:14s} tokens={s['n_tokens']:4d} scored={s['n_scored']:4d} mean_g={s['mean_g']:.3f} z={s['z']:6.2f}")
