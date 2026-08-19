
### Set `en`

| Variant | n | tokens | mean g | median z | detected (z≥2.33, 1 % FPR) | detected (z≥4) | similarity | surviving 5-grams |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No watermark (control) | 100 | 273 | 0.500 | 0.0 | 0% | 0% | — | — |
| Watermarked, untouched | 100 | 279 | 0.586 | 15.9 | 100% | 100% | 1.00 | 100% |
| Watermarked + Unicode cleaner | 100 | 279 | 0.586 | 15.9 | 100% | 100% | 1.00 | 100% |
| Delete 30 % of words | 100 | 195 | 0.521 | 3.0 | 72% | 32% | 0.91 | 27% |
| Round-trip translation via Spanish | 50 | 254 | 0.513 | 2.3 | 52% | 8% | 0.91 | 21% |
| Round-trip translation via Chinese | 50 | 265 | 0.508 | 1.2 | 20% | 2% | 0.91 | 13% |
| Paraphrase with another model | 100 | 209 | 0.501 | 0.1 | 2% | 0% | 0.84 | 2% |
| SIRA (70 % mask + fill-in) | 100 | 209 | 0.499 | -0.1 | 0% | 0% | 0.80 | 2% |

### Set `es`

| Variant | n | tokens | mean g | median z | detected (z≥2.33, 1 % FPR) | detected (z≥4) | similarity | surviving 5-grams |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No watermark (control) | 40 | 298 | 0.500 | -0.2 | 5% | 0% | — | — |
| Watermarked, untouched | 40 | 298 | 0.580 | 14.6 | 100% | 100% | 1.00 | 100% |
| Watermarked + Unicode cleaner | 40 | 298 | 0.580 | 14.6 | 100% | 100% | 1.00 | 100% |
| Delete 30 % of words | 40 | 206 | 0.526 | 3.7 | 92% | 45% | 0.91 | 36% |
| Round-trip translation via English | 40 | 291 | 0.509 | 1.8 | 28% | 5% | 0.92 | 18% |
| Round-trip translation via Chinese | 40 | 282 | 0.504 | 0.6 | 8% | 2% | 0.89 | 7% |
| Paraphrase with another model | 40 | 274 | 0.510 | 1.1 | 25% | 10% | 0.90 | 14% |
| SIRA (70 % mask + fill-in) | 40 | 291 | 0.508 | 1.1 | 22% | 10% | 0.85 | 12% |

### Set `long`

| Variant | n | tokens | mean g | median z | detected (z≥2.33, 1 % FPR) | detected (z≥4) | similarity | surviving 5-grams |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No watermark (control) | 20 | 988 | 0.499 | -0.5 | 0% | 0% | — | — |
| Watermarked, untouched | 20 | 993 | 0.607 | 37.8 | 100% | 100% | 1.00 | 100% |
| Watermarked + Unicode cleaner | 20 | 993 | 0.607 | 37.8 | 100% | 100% | 1.00 | 100% |
| Delete 30 % of words | 20 | 686 | 0.529 | 8.2 | 100% | 100% | 0.90 | 28% |
| Paraphrase with another model | 20 | 369 | 0.506 | 0.8 | 10% | 0% | 0.81 | 3% |
| SIRA (70 % mask + fill-in) | 20 | 294 | 0.503 | 0.8 | 10% | 0% | 0.74 | 2% |
| Paraphrase paragraph by paragraph | 20 | 1096 | 0.500 | 0.3 | 0% | 0% | 0.78 | 1% |
| SIRA paragraph by paragraph | 20 | 1527 | 0.500 | -0.2 | 5% | 0% | 0.75 | 1% |

### Set `low`

| Variant | n | tokens | mean g | median z | detected (z≥2.33, 1 % FPR) | detected (z≥4) | similarity | surviving 5-grams |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No watermark (control) | 10 | 283 | 0.503 | 0.2 | 0% | 0% | — | — |
| Watermarked, untouched | 10 | 267 | 0.537 | 6.0 | 90% | 70% | 1.00 | 100% |
| Watermarked + Unicode cleaner | 10 | 267 | 0.537 | 6.0 | 90% | 70% | 1.00 | 100% |
| Delete 30 % of words | 10 | 173 | 0.512 | 1.5 | 30% | 0% | 0.91 | 26% |
