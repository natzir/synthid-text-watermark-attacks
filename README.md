# SynthID-Text watermark attacks: replicating Claude's text watermark and breaking it on a laptop

Code and data behind the article **[Qué es, cómo funciona y cómo romper el watermark SynthID de Claude y Google](https://natzir.com/posicionamiento-buscadores/que-es-como-funciona-como-romper-synthid-watermark/)** (natzir.com, in Spanish).

On August 14, 2026 Anthropic announced that Claude will carry a statistical text watermark based on [SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4) (Google DeepMind, *Nature* 2024). Nobody outside Anthropic can measure that watermark (there is no key and no detection API yet), so this repo does what the research literature does: **replicate the mechanism with our own key and attack it**. The watermark here is the real SynthID-Text algorithm (the official implementation, as integrated in Hugging Face `transformers`) with **our** key; it is not Claude's or Google's watermark.

## What is in the repo

| Path | What it is |
|---|---|
| `common.py` | Configuration (key, 30 tournament layers, 4-token context), logits processors that force the hashing onto CPU, paths, helpers, `unicode_clean()` |
| `score.py` | Detector: the paper's *mean score* turned into a z-score (needs the key, not the model) |
| `prompts.py`, `prompts2.py` | The prompts (100 English blog-style, 40 Spanish, 20 long, 10 code/factual lists) |
| `gen2.py`, `attack2.py`, `attack_long_chunks.py` | **Run 2** (PyTorch + MPS): generator and attacker Qwen2.5-1.5B-Instruct |
| `mlxwm.py`, `gen3.py`, `attack3.py` | **Run 3** (MLX, the article's main run): generator Qwen2.5-7B-Instruct-4bit, attacker Llama-3.1-8B-Instruct-4bit |
| `analyze2.py`, `report2.py` | Scores every variant, computes detection rates, semantic similarity and surviving 5-grams; writes tables and SVG charts |
| `quality.py` | Per-response quality (perplexity under Llama 8B, LLM judge with position swap, lexical stats) |
| `diversity_gen.py`, `diversity_analyze.py` | Inter-response diversity for the same prompt (Self-BLEU, 4-gram overlap, embeddings) with and without watermark |
| `data/` | Generated and attacked texts of both runs (`gen*_<set>.jsonl`, `attacks*_<set>.jsonl`, `diversity.jsonl`) |
| `results/` | `results2.json` / `results3.json` (per item and summary), Markdown tables, SVG charts, `quality.json`, `diversity.json` |

Sets (`<set>`): `en` (100 prompts, ~300 tokens; round-trip translations on the first 50), `es` (40), `long` (20, ~1,000 tokens), `low` (10 code and factual-list prompts).

## Main results (run 3: 7B generator, Llama 8B attacker)

Detection = share of texts whose z-score exceeds the 1 % false-positive threshold (z ≥ 2.33). Full tables in `results/results3_table.md` and `results/results2_table.md`.

| Variant | English (n=100) | Spanish (n=40) | Long, ~1,000 tokens (n=20) |
|---|---:|---:|---:|
| No watermark (control) | 0 % | 5 % (2/40) | 0 % |
| Watermarked | 100 % (z 9.0) | 100 % (z 9.0) | 100 % (z 14.6) |
| + "Unicode cleaner" (what the removal skills do) | 100 % (identical) | 100 % | 100 % |
| Delete 30 % of the words at random | 41 % | 62 % | 80 % |
| Round-trip translation through a close language | 76 % (ES) | 80 % (EN) | — |
| Round-trip translation through Chinese | 16 % | 2 % | — |
| Paraphrase with Llama 8B | 5 % | 38 % | 5 % (paragraph by paragraph) |
| SIRA (70 % mask + guided fill-in) | 4 % | 30 % | 0 % |
| Code and factual lists, watermarked | 40 % (z 2.2) | | |

The share of the original's 5-token sequences that survive an attack predicts the z-score (r = 0.78 over 670 attacked texts). With a 1.5B generator (run 2) the mark is stronger (mean g-value 0.586 vs 0.547; z ≈ 16 vs ≈ 9): the better the model, the lower the entropy and the weaker the mark.

Quality (`results/quality.json`, `results/diversity.json`): per response there is no measurable difference between watermarked and unwatermarked text (perplexity, LLM judge, lexical stats); across responses to the same prompt the watermark reduces diversity a lot (Self-BLEU 38.6 vs 19.4 at temperature 1; 45.8 vs 28.6 at 0.7), as the Nature paper's supplement acknowledges.

## Install

Tested on a MacBook Pro M1 (16 GB) with Python 3.10. Models are downloaded from Hugging Face on first use (about 9 GB for the MLX run).

```bash
pip install -r requirements.txt
```

## Reproduce

All scripts read from `data/` and write to `data/` and `results/`. Point `WM_DATA_DIR` and `WM_RESULTS_DIR` somewhere else to avoid overwriting the published data.

```bash
# Run 3 (MLX, Apple Silicon): watermarked/unwatermarked generation, then attacks
python gen3.py en        # also: es, long, low;  add a number to limit prompts (smoke test)
python attack3.py en     # paraphrase, SIRA and round-trip translations (translations only on the first 50 of `en`)
RUN=3 python analyze2.py && RUN=3 python report2.py     # -> results/results3*.{json,md}, results/chart3_*.svg

# Run 2 (PyTorch/MPS, Qwen2.5-1.5B as generator and attacker)
python gen2.py en 8 && python attack2.py en 8           # batch 8; LIMIT=2 for a smoke test
python gen2.py long 4 && python attack_long_chunks.py 8 # long texts, paragraph by paragraph
python analyze2.py && python report2.py                 # -> results/results2*.{json,md}, results/chart2_*.svg

# Quality and diversity (on run 3)
python quality.py en es
python diversity_gen.py 30 5 && python diversity_analyze.py

# Score any jsonl by hand
python score.py data/gen3_en.jsonl
```

`ATTACK_MODEL_OVERRIDE=<HF model id>` changes the run-2 attacker (default Qwen2.5-1.5B-Instruct; Qwen2.5-3B-Instruct swapped on 16 GB when batched).

## Two technical details worth knowing

- **Apple Silicon (MPS)**: the g-value hashing of the Hugging Face implementation does not match between MPS and CPU. Generate with the processor on MPS and detect on CPU and the watermark "disappears" (mean g-value 0.50). `common.py` therefore forces the processor onto CPU (`CPUSynthID`), and `mlxwm.py` plugs it into MLX as a logits processor.
- **Top-k order**: HF applies custom logits processors *before* its top-k warper, while the official `watermarking_config` path applies the watermark *after* top-k. `FastCPUSynthID` applies top-k first and computes g-values only for the 40 candidates (15x faster, token-identical output to the reference processor).

## Limits

The "Claude" stand-in is a quantized 7B, not a frontier model (which would have less entropy and a weaker mark). The detector is the paper's simple mean score; Anthropic can use the Bayesian detector and other parameters, which recovers some signal after an attack. SIRA's fill-in step with small models sometimes loses meaning. Semantic similarity uses a sentence-transformer that truncates at ~128 tokens (indicative only for long texts). Above all, **none of this is measured against Claude's real watermark**, because nobody can measure it yet; what is identical is the mechanism.

## References

- Dathathri et al., *Scalable watermarking for identifying large language model outputs*, Nature 2024 ([paper](https://www.nature.com/articles/s41586-024-08025-4), [code](https://github.com/google-deepmind/synthid-text)).
- Cheng et al., *Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks* (SIRA), ICML 2025 ([arXiv](https://arxiv.org/abs/2505.05190), [code](https://github.com/Allencheng97/Self-information-Rewrite-Attack)); the paraphrase and fill-in prompts are the ones from the SIRA repo.
- Anthropic, *How Claude's text watermark works*, Aug 14, 2026 ([post](https://www.anthropic.com/news/claude-text-watermark)).

## License

MIT. Author: Natzir ([natzir.com](https://natzir.com) · [LinkedIn](https://www.linkedin.com/in/natzir/) · [X/Twitter](https://x.com/natzir9)).
