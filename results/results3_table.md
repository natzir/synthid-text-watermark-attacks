
### Set `en`

| Variant | n | tokens | mean g | median z | detected (z≥2.33, 1 % FPR) | detected (z≥4) | similarity | surviving 5-grams |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No watermark (control) | 100 | 297 | 0.499 | -0.2 | 0% | 0% | — | — |
| Watermarked, untouched | 100 | 299 | 0.547 | 9.0 | 100% | 97% | 1.00 | 100% |
| Watermarked + Unicode cleaner | 100 | 299 | 0.547 | 9.0 | 100% | 97% | 1.00 | 100% |
| Delete 30 % of words | 100 | 209 | 0.513 | 2.1 | 41% | 9% | 0.91 | 29% |
| Round-trip translation via Spanish | 50 | 309 | 0.519 | 3.6 | 76% | 36% | 0.97 | 50% |
| Round-trip translation via Chinese | 50 | 308 | 0.505 | 0.6 | 16% | 8% | 0.90 | 14% |
| Paraphrase with another model | 100 | 324 | 0.503 | 0.5 | 5% | 1% | 0.84 | 8% |
| SIRA (70 % mask + fill-in) | 100 | 277 | 0.503 | 0.7 | 4% | 0% | 0.83 | 8% |

### Set `es`

| Variant | n | tokens | mean g | median z | detected (z≥2.33, 1 % FPR) | detected (z≥4) | similarity | surviving 5-grams |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No watermark (control) | 40 | 300 | 0.498 | -0.3 | 5% | 0% | — | — |
| Watermarked, untouched | 40 | 298 | 0.550 | 9.0 | 100% | 100% | 1.00 | 100% |
| Watermarked + Unicode cleaner | 40 | 298 | 0.550 | 9.0 | 100% | 100% | 1.00 | 100% |
| Delete 30 % of words | 40 | 207 | 0.518 | 2.8 | 62% | 10% | 0.92 | 39% |
| Round-trip translation via English | 40 | 312 | 0.522 | 3.7 | 80% | 45% | 0.96 | 50% |
| Round-trip translation via Chinese | 40 | 328 | 0.503 | 0.4 | 2% | 0% | 0.87 | 11% |
| Paraphrase with another model | 40 | 307 | 0.509 | 1.5 | 38% | 12% | 0.87 | 21% |
| SIRA (70 % mask + fill-in) | 40 | 306 | 0.508 | 1.3 | 30% | 10% | 0.87 | 19% |

### Set `long`

| Variant | n | tokens | mean g | median z | detected (z≥2.33, 1 % FPR) | detected (z≥4) | similarity | surviving 5-grams |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No watermark (control) | 20 | 1038 | 0.500 | 0.0 | 0% | 0% | — | — |
| Watermarked, untouched | 20 | 1029 | 0.544 | 14.6 | 100% | 100% | 1.00 | 100% |
| Watermarked + Unicode cleaner | 20 | 1029 | 0.544 | 14.6 | 100% | 100% | 1.00 | 100% |
| Delete 30 % of words | 20 | 710 | 0.513 | 3.5 | 80% | 35% | 0.92 | 28% |
| Paraphrase paragraph by paragraph | 20 | 896 | 0.501 | 0.3 | 5% | 0% | 0.75 | 1% |
| SIRA paragraph by paragraph | 20 | 938 | 0.501 | 0.3 | 0% | 0% | 0.79 | 3% |

### Set `low`

| Variant | n | tokens | mean g | median z | detected (z≥2.33, 1 % FPR) | detected (z≥4) | similarity | surviving 5-grams |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No watermark (control) | 10 | 285 | 0.503 | 0.4 | 0% | 0% | — | — |
| Watermarked, untouched | 10 | 285 | 0.513 | 2.2 | 40% | 10% | 1.00 | 100% |
| Watermarked + Unicode cleaner | 10 | 285 | 0.513 | 2.2 | 40% | 10% | 1.00 | 100% |
| Delete 30 % of words | 10 | 191 | 0.504 | 0.6 | 0% | 0% | 0.93 | 26% |
