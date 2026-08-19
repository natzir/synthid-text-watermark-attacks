"""Run 3 (MLX): generation with Qwen2.5-7B-Instruct-4bit. Usage: python3 gen3.py <set> [limit]  -> data/gen3_<set>.jsonl"""
import sys, time
from mlx_lm import load
from common import dump_jsonl, DATA
from prompts2 import SETS
from mlxwm import GEN_MLX, chat_prompt, gen_text

SET = sys.argv[1]; LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
cfg = SETS[SET]
model, tok = load(GEN_MLX)
system = "Eres un asistente útil. Responde siempre en español." if cfg["lang"] == "es" else "You are a helpful assistant."
prompts = cfg["prompts"][:LIMIT] if LIMIT else cfg["prompts"]
rows, t0 = [], time.time()
for i, p in enumerate(prompts):
    pr = chat_prompt(tok, p, system)
    row = {"set": SET, "idx": i, "prompt": p,
           "wm": gen_text(model, tok, pr, cfg["max_new"], True, seed=7000 + i),
           "nowm": gen_text(model, tok, pr, cfg["max_new"], False, seed=7000 + i)}
    rows.append(row); dump_jsonl(DATA / f"gen3_{SET}.jsonl", rows)
    print(f"[{i+1}/{len(prompts)}] {SET} wm={len(row['wm'].split())}w nowm={len(row['nowm'].split())}w  {time.time()-t0:.0f}s", flush=True)
print("done", SET, time.time() - t0)
