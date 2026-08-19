"""SVG bar charts (median z per variant), one per set. RUN="" -> results/results2.json, RUN=3 -> results/results3.json."""
import json, html, os
from common import RESULTS
RUN = os.environ.get("RUN", "")
R = json.load(open(RESULTS / f"results{RUN or 2}.json"))["summary"]
LABELS = {
    "nowm": "Sin marca (control)", "wm": "Con marca, sin tocar", "wm_unicode_cleaner": "+ limpiador Unicode (\"skill\")",
    "wm_word_delete30": "Borrar 30 % de palabras", "paraphrase": "Paráfrasis con otro modelo",
    "rt_es": "Ida-vuelta por español", "rt_en": "Ida-vuelta por inglés", "rt_zh": "Ida-vuelta por chino",
    "sira": "SIRA (máscara 70 % + relleno)", "paraphrase_ch": "Paráfrasis párrafo a párrafo", "sira_ch": "SIRA párrafo a párrafo",
}
ORDER = ["nowm", "wm", "wm_unicode_cleaner", "wm_word_delete30", "rt_es", "rt_en", "rt_zh", "paraphrase", "sira", "paraphrase_ch", "sira_ch"]

def chart(S, title):
    keys = [f"{S}:{v}" for v in ORDER if f"{S}:{v}" in R]
    if not keys: return None
    W, H_ROW, PAD_L, PAD_R, PAD_T, PAD_B = 720, 32, 250, 70, 34, 30
    H = PAD_T + PAD_B + H_ROW * len(keys)
    zmax = max(R[k]["z_median"] for k in keys); xmax = max(zmax, 4.5) * 1.1
    x0 = PAD_L
    sx = lambda z: x0 + max(0.0, z) / xmax * (W - PAD_L - PAD_R)
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(title)}" xmlns="http://www.w3.org/2000/svg" style="font-family:inherit;max-width:100%">']
    step = 5 if xmax < 30 else 10
    g = 0
    while g <= xmax:
        p.append(f'<line x1="{sx(g):.1f}" y1="{PAD_T-6}" x2="{sx(g):.1f}" y2="{H-PAD_B+4}" stroke="var(--grid)" stroke-width="1"/>')
        p.append(f'<text x="{sx(g):.1f}" y="{H-PAD_B+18}" text-anchor="middle" font-size="11" fill="var(--text-2)">{g}</text>')
        g += step
    for tz, lab in [(2.326, "z=2,33 (1 % FP)"), (4.0, "z=4")]:
        p.append(f'<line x1="{sx(tz):.1f}" y1="{PAD_T-10}" x2="{sx(tz):.1f}" y2="{H-PAD_B+4}" stroke="var(--text-2)" stroke-width="1.5" stroke-dasharray="4 3"/>')
        p.append(f'<text x="{sx(tz)+4:.1f}" y="{PAD_T-14}" font-size="11" fill="var(--text-2)">{lab}</text>')
    for i, k in enumerate(keys):
        y = PAD_T + i * H_ROW + 7; r = R[k]; z = r["z_median"]; x1 = sx(z)
        p.append(f'<text x="{PAD_L-10}" y="{y+13}" text-anchor="end" font-size="12.5" fill="var(--text-1)">{html.escape(LABELS[r["variant"]])}</text>')
        if z > 0:
            p.append(f'<rect x="{x0:.1f}" y="{y}" width="{max(2.0, x1-x0):.1f}" height="18" fill="var(--bar)"/>')
        det = f'{r["det_rate_1pct"]:.0%}'.replace("%", " %")
        p.append(f'<text x="{max(x1, x0)+8:.1f}" y="{y+13}" font-size="12" fill="var(--text-1)" font-variant-numeric="tabular-nums">{z:.1f} · {det} det.</text>')
    p.append(f'<line x1="{x0}" y1="{PAD_T-6}" x2="{x0}" y2="{H-PAD_B+4}" stroke="var(--text-2)" stroke-width="1"/>')
    p.append("</svg>")
    return "\n".join(p)

for S, title in [("en", "Inglés, ~280 tokens (n=100; traducciones n=50)"), ("es", "Español, ~300 tokens (n=40)"), ("long", "Inglés largo, ~1.000 tokens (n=20)")]:
    svg = chart(S, title)
    if svg:
        open(RESULTS / f"chart{RUN or 2}_{S}.svg", "w").write(svg); print("wrote", S)
