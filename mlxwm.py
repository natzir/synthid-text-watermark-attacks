"""SynthID-Text on MLX: reuse the reference (HF, CPU) processor maths inside mlx_lm's generation loop."""
import numpy as np, torch
import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler
from common import fresh_fast_processor, chat as chat_prompt

__all__ = ["GEN_MLX", "ATTACK_MLX", "MLXSynthID", "chat_prompt", "gen_text"]

GEN_MLX = "mlx-community/Qwen2.5-7B-Instruct-4bit"          # same tokenizer as Qwen2.5-1.5B -> score.py unchanged
ATTACK_MLX = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"  # SIRA's "Small" attack model family

class MLXSynthID:
    """mlx_lm logits processor: (tokens[1-D], logits[1,V]) -> logits. Fresh HF processor per generation."""
    def __init__(self, top_k=40):
        self.proc = fresh_fast_processor(top_k)
    def __call__(self, tokens, logits):
        ids = torch.from_numpy(np.array(tokens, dtype=np.int64))[None, :]
        sc = torch.from_numpy(np.array(logits.astype(mx.float32), copy=False))
        out = self.proc(ids, sc)
        return mx.array(out.numpy())

def gen_text(model, tok, prompt, max_tokens, watermark, seed, temp=1.0, top_k=40, greedy=False):
    mx.random.seed(seed)
    sampler = make_sampler(temp=0.0) if greedy else make_sampler(temp=temp, top_k=top_k)
    procs = [MLXSynthID(top_k)] if watermark else None
    return generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler, logits_processors=procs, verbose=False)
