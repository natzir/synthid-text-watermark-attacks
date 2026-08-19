"""Shared configuration: SynthID-Text parameters (our own key), CPU-side logits processors, paths and helpers.

Two runs live in this repo:
  * run "2" (PyTorch + MPS): generator and attacker Qwen2.5-1.5B-Instruct      -> data/gen_*.jsonl, data/attacks_*.jsonl
  * run "3" (MLX):           generator Qwen2.5-7B-Instruct-4bit, attacker Llama-3.1-8B-Instruct-4bit
                                                                                 -> data/gen3_*.jsonl, data/attacks3_*.jsonl
The detector (score.py) only needs a tokenizer and the key; all Qwen2.5 sizes share the vocabulary, so the same detector
scores both runs.
"""
import json, os
from pathlib import Path
import torch
from transformers import SynthIDTextWatermarkLogitsProcessor

ROOT = Path(__file__).resolve().parent
# Override with WM_DATA_DIR / WM_RESULTS_DIR (e.g. for a smoke test) so you never clobber the published data.
DATA = Path(os.environ.get("WM_DATA_DIR", ROOT / "data"))
RESULTS = Path(os.environ.get("WM_RESULTS_DIR", ROOT / "results"))
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# PyTorch run. Qwen2.5-3B-Instruct as attacker thrashed swap on a 16 GB M1 Pro when batched; the published
# attacks_*.jsonl were produced with the 1.5B. Override with the env var ATTACK_MODEL_OVERRIDE if you have more memory.
GEN_MODEL_TORCH = "Qwen/Qwen2.5-1.5B-Instruct"
ATTACK_MODEL_TORCH = "Qwen/Qwen2.5-1.5B-Instruct"
# Tokenizer used by the detector and for n-gram overlap (shared by every Qwen2.5 size, including the 7B MLX generator).
TOKENIZER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# SynthID-Text configuration: 30 keys = tournament depth 30 (the Nature paper's production setting);
# ngram_len 5 = the random seed is a hash of the 4 previous tokens plus the key. The key is public here on purpose:
# this is a replica of the mechanism with our own key, not Anthropic's or Google's watermark.
KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468, 990,
        966, 226, 324, 585, 118, 504, 421, 521, 129, 669, 732, 225, 90, 960]
NGRAM_LEN = 5
SAMPLING_TABLE_SIZE = 65536
SAMPLING_TABLE_SEED = 0
CONTEXT_HISTORY_SIZE = 1024


def wm_processor(device="cpu"):
    """Reference Hugging Face processor; used for detection (compute_g_values / repetition mask)."""
    return SynthIDTextWatermarkLogitsProcessor(
        ngram_len=NGRAM_LEN, keys=KEYS, sampling_table_size=SAMPLING_TABLE_SIZE,
        sampling_table_seed=SAMPLING_TABLE_SEED, context_history_size=CONTEXT_HISTORY_SIZE, device=device)


class CPUSynthID(SynthIDTextWatermarkLogitsProcessor):
    """Runs the SynthID hashing on CPU even when the model runs on MPS: on Apple Silicon the int64 hashing path
    produces g-values that do NOT match the reference (CPU) implementation, so generation and detection must both
    use the CPU hash or the "watermark" is silently wrong (mean g-value stays at 0.5). One instance per generate()
    call: the processor keeps state."""
    def __call__(self, input_ids, scores):
        out = super().__call__(input_ids.to("cpu"), scores.float().to("cpu"))
        return out.to(scores.device, dtype=scores.dtype)


class FastCPUSynthID(CPUSynthID):
    """Same maths as the reference processor, but g-values are computed only for the top-k candidates.
    This is exact: tokens with -inf score have zero probability in update_scores, so they cannot change the result.
    Top-k is applied here first because HF merges custom processors *before* its top-k warper, while the reference
    `watermarking_config` path runs SynthID *after* TopKLogitsWarper; the later warper is then a no-op.
    About 15x faster than the reference on CPU. Verified to produce token-identical outputs to CPUSynthID."""
    top_k = 40

    def __call__(self, input_ids, scores):
        input_ids = input_ids.to("cpu")
        s = scores.float().to("cpu")
        batch_size, _ = s.shape
        if self.state is None:
            self._init_state(batch_size)
        else:
            self.state.context = torch.concat((self.state.context, input_ids[:, -1:]), dim=1)[:, 1:]
        self.state.num_calls += 1
        if self.skip_first_ngram_calls and self.state.num_calls < self.ngram_len:
            return scores
        k = min(self.top_k, int(torch.isfinite(s).sum(dim=1).max().clamp(min=1)))
        vals, idx = torch.topk(s, k, dim=1)
        s = torch.full_like(s, torch.finfo(s.dtype).min)  # everything outside top-k is masked out
        ngram_keys, ctx_hash = self._compute_keys(self.state.context, idx)
        g = self.sample_g_values(ngram_keys)
        upd = self.update_scores(vals, g)
        ctx_hash = ctx_hash[:, None]
        is_rep = (self.state.context_history == ctx_hash).any(dim=1, keepdim=True)
        self.state.context_history = torch.concat((ctx_hash, self.state.context_history), dim=1)[:, :-1]
        new_vals = torch.where(is_rep, vals, upd)
        out = s.clone()
        out.scatter_(1, idx, new_vals)
        return out.to(scores.device, dtype=scores.dtype)


def fresh_fast_processor(top_k=40):
    p = FastCPUSynthID(ngram_len=NGRAM_LEN, keys=KEYS, sampling_table_size=SAMPLING_TABLE_SIZE,
                       sampling_table_seed=SAMPLING_TABLE_SEED, context_history_size=CONTEXT_HISTORY_SIZE, device="cpu")
    p.top_k = top_k
    return p


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def dump_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def chat(tok, user, system="You are a helpful assistant."):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


INVISIBLE_CHARS = ("\u200b\u200c\u200d\u2060\ufeff\u00ad"  # zero-width space/non-joiner/joiner, word joiner, BOM, soft hyphen
                   "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")  # bidi marks and isolates


def unicode_clean(text):
    """What the "watermark cleaner" skills do: strip zero-width / bidi / BOM / soft-hyphen characters and normalise
    the non-breaking spaces. A sampling watermark adds none of these, so the z-score is unchanged."""
    text = text.translate({ord(c): None for c in INVISIBLE_CHARS})
    return text.replace("\u202f", " ").replace("\u00a0", " ")  # narrow/regular no-break space -> space
