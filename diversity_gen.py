"""Diversity across responses: k samples per prompt, with and without watermark, at two temperatures.
Nature SI G.3 / Extended Data Fig. 4: single-sequence non-distortion does not preserve inter-response diversity."""
import sys, time
from mlx_lm import load
from common import dump_jsonl, DATA
from prompts import GENERAL
from mlxwm import GEN_MLX, chat_prompt, gen_text

N_PROMPTS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TEMPS = [1.0, 0.7]
model, tok = load(GEN_MLX)
rows, t0 = [], time.time()
for i, p in enumerate(GENERAL[:N_PROMPTS]):
    pr = chat_prompt(tok, p)
    row = {"idx": i, "prompt": p}
    for T in TEMPS:
        for wm in (True, False):
            key = f"{'wm' if wm else 'nowm'}_T{T}"
            row[key] = [gen_text(model, tok, pr, 250, wm, seed=9000 + 100 * i + j, temp=T) for j in range(K)]
    rows.append(row); dump_jsonl(DATA / "diversity.jsonl", rows)
    print(f"[{i+1}/{N_PROMPTS}] {time.time()-t0:.0f}s", flush=True)
print("done", time.time() - t0)
