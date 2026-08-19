"""Run 2 (PyTorch/MPS, Qwen2.5-1.5B): batched generation. Usage: python3 gen2.py <set> [batch_size]
Writes data/gen_<set>.jsonl with, per prompt: wm (SynthID, top-k 40, 30 layers) and nowm (same sampling, same seed).
Set LIMIT=<n> to generate only the first n prompts (smoke test)."""
import os, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from common import GEN_MODEL_TORCH, DATA, device, chat, fresh_fast_processor, dump_jsonl
from prompts2 import SETS

SET = sys.argv[1]
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
cfg = SETS[SET]
dev = device()
tok = AutoTokenizer.from_pretrained(GEN_MODEL_TORCH, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_TORCH, dtype=torch.float16).to(dev).eval()
system = "Eres un asistente útil. Responde siempre en español." if cfg["lang"] == "es" else "You are a helpful assistant."

def gen_batch(prompts, watermark, seed):
    torch.manual_seed(seed)
    enc = tok([chat(tok, p, system) for p in prompts], return_tensors="pt", padding=True).to(dev)
    kw = dict(do_sample=True, temperature=1.0, top_k=40, max_new_tokens=cfg["max_new"], pad_token_id=tok.eos_token_id)
    if watermark:
        kw["logits_processor"] = LogitsProcessorList([fresh_fast_processor(40)])
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    new = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(x, skip_special_tokens=True) for x in new]

prompts = cfg["prompts"][:int(os.environ["LIMIT"])] if os.environ.get("LIMIT") else cfg["prompts"]
rows, t0 = [], time.time()
for b in range(0, len(prompts), BS):
    chunk = prompts[b:b + BS]
    wm = gen_batch(chunk, True, seed=5000 + b)
    nowm = gen_batch(chunk, False, seed=5000 + b)
    for j, p in enumerate(chunk):
        rows.append({"set": SET, "idx": b + j, "prompt": p, "wm": wm[j], "nowm": nowm[j]})
    dump_jsonl(DATA / f"gen_{SET}.jsonl", rows)
    print(f"[{len(rows)}/{len(prompts)}] {SET} wm_words={sum(len(x.split()) for x in wm)//len(wm)} "
          f"nowm_words={sum(len(x.split()) for x in nowm)//len(nowm)}  {time.time()-t0:.0f}s", flush=True)
print("done", SET, time.time() - t0)
