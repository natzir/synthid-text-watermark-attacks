"""SVG bar charts (median z-score per variant, English labels), one per set. RUN="" -> results/results2.json, RUN=3 -> results/results3.json."""
import json, html, os
from common import RESULTS
RUN = os.environ.get("RUN", "")
R = json.load(open(RESULTS / f"results{RUN or 2}.json"))["summary"]
LABELS = {
    "nowm": "No watermark (control)", "wm": "Watermarked, untouched", "wm_unicode_cleaner": "+ Unicode cleaner (\"skill\")",
    "wm_word_delete30": "Delete 30 % of words", "paraphrase": "Paraphrase with another model",
    "rt_es": "Round-trip via Spanish", "rt_en": "Round-trip via English", "rt_zh": "Round-trip via Chinese",
    "sira": "SIRA (70 % mask + fill-in)", "paraphrase_ch": "Paraphrase paragraph by paragraph", "sira_ch": "SIRA paragraph by paragraph",
}
ORDER = ["nowm", "wm", "wm_unicode_cleaner", "wm_word_delete30", "rt_es", "rt_en", "rt_zh", "paraphrase", "sira", "paraphrase_ch", "sira_ch"]

def chart(S, title):
    keys = [f"{S}:{v}" for v in ORDER if f"{S}:{v}" in R]
    if not keys: return None
    W, H_ROW, PAD_L, PAD_R, PAD_T, PAD_B = 820, 32, 250, 190, 70, 34
    H = PAD_T + PAD_B + H_ROW * len(keys)
    zmax = max(R[k]["z_median"] for k in keys); xmax = max(zmax, 4.5) * 1.08
    x0 = PAD_L
    sx = lambda z: x0 + max(0.0, z) / xmax * (W - PAD_L - PAD_R)
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(title)}" xmlns="http://www.w3.org/2000/svg" style="font-family:inherit;max-width:100%">']
    p.append(f'<text x="{PAD_L}" y="18" font-size="13" font-weight="600" fill="var(--text-1)">{html.escape(title)}</text>')
    p.append(f'<text x="{PAD_L}" y="34" font-size="11" fill="var(--text-2)">Median z-score per variant · label = z · share detected at 1 % FPR (z ≥ 2.33)</text>')
    step = 5 if xmax < 30 else 10
    g = 0
    while g <= xmax:
        p.append(f'<line x1="{sx(g):.1f}" y1="{PAD_T-6}" x2="{sx(g):.1f}" y2="{H-PAD_B+4}" stroke="var(--grid)" stroke-width="1"/>')
        p.append(f'<text x="{sx(g):.1f}" y="{H-PAD_B+18}" text-anchor="middle" font-size="11" fill="var(--text-2)">{g}</text>')
        g += step
    # thresholds: the z=2.33 label sits left of its line, the z=4 label right of its line, so they never collide
    for tz, lab, anchor, dx in [(2.326, "z = 2.33 (1 % FPR)", "end", -4), (4.0, "z = 4", "start", 4)]:
        p.append(f'<line x1="{sx(tz):.1f}" y1="{PAD_T-10}" x2="{sx(tz):.1f}" y2="{H-PAD_B+4}" stroke="var(--text-2)" stroke-width="1.5" stroke-dasharray="4 3"/>')
        p.append(f'<text x="{sx(tz)+dx:.1f}" y="{PAD_T-16}" text-anchor="{anchor}" font-size="11" fill="var(--text-2)">{lab}</text>')
    for i, k in enumerate(keys):
        y = PAD_T + i * H_ROW + 7; r = R[k]; z = r["z_median"]; x1 = sx(z)
        p.append(f'<text x="{PAD_L-10}" y="{y+13}" text-anchor="end" font-size="12.5" fill="var(--text-1)">{html.escape(LABELS[r["variant"]])}</text>')
        if z > 0:
            p.append(f'<rect x="{x0:.1f}" y="{y}" width="{max(2.0, x1-x0):.1f}" height="18" fill="var(--bar)"/>')
        det = f'{r["det_rate_1pct"]:.0%}'.replace("%", " %")
        p.append(f'<text x="{max(x1, x0)+8:.1f}" y="{y+13}" font-size="12" fill="var(--text-1)" font-variant-numeric="tabular-nums">{z:.1f} · {det} detected</text>')
    p.append(f'<line x1="{x0}" y1="{PAD_T-6}" x2="{x0}" y2="{H-PAD_B+4}" stroke="var(--text-2)" stroke-width="1"/>')
    p.append("</svg>")
    return "\n".join(p)

for S, title in [("en", "English, ~300 tokens (n=100; translations n=50)"), ("es", "Spanish, ~300 tokens (n=40)"), ("long", "English, long ~1,000 tokens (n=20)")]:
    svg = chart(S, title)
    if svg:
        open(RESULTS / f"chart{RUN or 2}_{S}.svg", "w").write(svg); print("wrote", S)
