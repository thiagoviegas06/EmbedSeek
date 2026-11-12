"""Method overview figure (schematic, no data): EmbedSeek's proposed two-level bandit.

Usage: python -I scripts/make_method_figure.py docs/figures/method_overview.svg"""
import random
import sys

W, H = 1400, 790
SURF, INK, INK2, MUTED, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#c9c8c2"
FIX_FILL, FIX_STROKE = "#f1f0ec", "#8a8984"          # fixed for every method
VAR_FILL, VAR_STROKE = "#e6f0fc", "#2a78d6"          # research variable
LLM_FILL, LLM_STROKE = "#fdeee7", "#eb6834"          # expensive oracle / LLM
POS, NEG = "#1baf7a", "#b9b8b2"

out = []
add = out.append


def text(x, y, s, size=13, weight="normal", fill=INK, anchor="start", style="normal"):
    add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}" font-style="{style}">{s}</text>')


def box(x, y, w, h, fill, stroke, dashed=False, rx=8):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')


def labeled_box(x, y, w, h, lines, fill, stroke, dashed=False, first_bold=True):
    box(x, y, w, h, fill, stroke, dashed)
    ty = y + 22
    for i, s in enumerate(lines):
        text(x + 14, ty, s, size=14 if i == 0 else 12, weight="bold" if i == 0 and first_bold else "normal",
             fill=INK if i == 0 else INK2)
        ty += 19


def arrow(d, color=INK2, dashed=False, width=1.6):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    marker = "ahd" if dashed else "ah"
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash} marker-end="url(#{marker})"/>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    'font-family="-apple-system, Helvetica Neue, Helvetica, Arial, sans-serif">')
add('<defs>'
    f'<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker>'
    f'<marker id="ahd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{VAR_STROKE}"/></marker>'
    '</defs>')
add(f'<rect width="{W}" height="{H}" fill="{SURF}"/>')

text(24, 38, "EmbedSeek (proposed method): search several embedding spaces with a two-level bandit, then train one fixed proxy",
     size=18, weight="bold")

# Panels
for x, w, title in [(20, 425, "(a) Embed, screen, cluster"), (465, 545, "(b) Two-level bandit chooses what to label"),
                    (1030, 350, "(c) Proxy: identical for every method")]:
    add(f'<rect x="{x}" y="62" width="{w}" height="640" rx="12" fill="none" stroke="{LINE}" stroke-width="1.2"/>')
    text(x + 16, 90, title, size=15, weight="bold")

# ---------- Panel (a) ----------
labeled_box(48, 108, 370, 54, ["Table to filter: N rows", "e.g. 143,354 Wikipedia talk-page comments"], FIX_FILL, FIX_STROKE)
arrow("M233,162 V186")
labeled_box(48, 188, 370, 54, ["Proxy embedding (MiniLM), all N rows", "needed anyway: the proxy must score every row"], FIX_FILL, FIX_STROKE)
arrow("M233,242 V266")
labeled_box(48, 268, 370, 54, ["Screen (optional): top-s rows by query", "similarity ∪ a random slice → cuts encoder cost"], FIX_FILL, FIX_STROKE, dashed=True)

enc_x = [48, 178, 308]
enc_names = [("Encoder 1", "e.g. mpnet"), ("Encoder 2", "e.g. bge"), ("Encoder M", "…")]
for x, (a, b) in zip(enc_x, enc_names):
    arrow(f"M233,322 L{x + 55},356")
    box(x, 358, 110, 48, VAR_FILL, VAR_STROKE)
    text(x + 55, 378, a, size=13, weight="bold", anchor="middle")
    text(x + 55, 396, b, size=11.5, fill=INK2, anchor="middle")
    arrow(f"M{x + 55},406 V424")

# Cluster scatter per encoder: same rows, different geometry -> positives land in different clusters.
rng = random.Random(7)
layouts = [  # (cx, cy, rx, ry, share of positives) relative to the panel box
    [(30, 35, 22, 17, 0.7), (78, 40, 22, 18, 0.0), (52, 88, 30, 18, 0.15)],
    [(32, 85, 24, 20, 0.0), (80, 80, 22, 22, 0.55), (50, 34, 30, 16, 0.3)],
    [(55, 40, 34, 18, 0.05), (28, 88, 22, 18, 0.35), (82, 90, 20, 18, 0.35)],
]
for x, layout in zip(enc_x, layouts):
    box(x, 426, 110, 120, "#ffffff", VAR_STROKE, rx=6)
    for cx, cy, rx, ry, pshare in layout:
        add(f'<ellipse cx="{x + cx}" cy="{426 + cy}" rx="{rx}" ry="{ry}" fill="none" stroke="{VAR_STROKE}" '
            'stroke-width="1" stroke-dasharray="3 2" opacity="0.8"/>')
        for _ in range(9):
            a, r = rng.uniform(0, 6.283), rng.uniform(0, 0.8) ** 0.5
            px = x + cx + r * (rx - 4) * __import__("math").cos(a)
            py = 426 + cy + r * (ry - 4) * __import__("math").sin(a)
            color = POS if rng.random() < pshare else NEG
            add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="{color}"/>')
text(233, 568, "k-means clusters inside each encoder", size=12.5, fill=INK2, anchor="middle")
text(233, 586, "= the arms of the second level", size=12.5, fill=INK2, anchor="middle")
text(233, 630, "Same rows, different geometry: each encoder", size=12.5, fill=INK, anchor="middle")
text(233, 648, "may group the positives differently.", size=12.5, fill=INK, anchor="middle")
text(233, 682, "Embedding cost = C_proxy(N) + Σₘ s · C_m(N)", size=12.5, fill=INK2, anchor="middle", style="italic")

# ---------- Panel (b) ----------
BX, BW = 492, 430


def bars(x0, base, heights, pick, width, gap, labels):
    for i, (h, lab) in enumerate(zip(heights, labels)):
        x = x0 + i * (width + gap)
        fill = VAR_STROKE if i == pick else "#b7d3f6"
        add(f'<rect x="{x}" y="{base - h}" width="{width}" height="{h}" rx="3" fill="{fill}"/>')
        text(x + width / 2, base + 14, lab, size=11, fill=INK2, anchor="middle")
    add(f'<line x1="{x0 - 4}" y1="{base}" x2="{x0 + len(heights) * (width + gap) - gap + 4}" y2="{base}" stroke="{INK2}" stroke-width="1"/>')


box(BX, 108, BW, 104, VAR_FILL, VAR_STROKE)
text(BX + 14, 132, "① Pick an encoder", size=14, weight="bold")
for i, s in enumerate(["Thompson sampling: draw a score from each", "encoder's Beta(wins, losses); take the largest", "(top level: one arm per embedding space)"]):
    text(BX + 14, 153 + 18 * i, s, size=12, fill=INK2)
bars(BX + 318, 190, [28, 56, 38], 1, 24, 14, ["E1", "E2", "EM"])

arrow(f"M{BX + BW / 2},212 V236")
box(BX, 238, BW, 104, VAR_FILL, VAR_STROKE)
text(BX + 14, 262, "② Pick a cluster inside it", size=14, weight="bold")
for i, s in enumerate(["Same Thompson sampling over the chosen", "encoder's clusters (this level is LTS-lite)", "(second level: one arm per cluster)"]):
    text(BX + 14, 283 + 18 * i, s, size=12, fill=INK2)
bars(BX + 306, 320, [20, 14, 50, 26, 10], 2, 16, 7, ["c1", "c2", "c3", "c4", "c5"])

arrow(f"M{BX + BW / 2},342 V366")
labeled_box(BX, 368, BW, 64, ["③ Propose a batch of b rows from that cluster",
                              "likely positives first (LR in that encoder's space), rest random"], VAR_FILL, VAR_STROKE)
arrow(f"M{BX + BW / 2},432 V456")
labeled_box(BX, 458, BW, 64, ["④ Oracle labels the batch",
                              "human labels now, an LLM later: the expensive step"], LLM_FILL, LLM_STROKE)
arrow(f"M{BX + BW / 2},522 V546")
labeled_box(BX, 548, BW, 82, ["⑤ Reward and update",
                              "r = 1 if the batch contains a positive; update the",
                              "cluster and the encoder; decay old counts (δ ≈ 0.99)"], VAR_FILL, VAR_STROKE)

# Loop back to step 1, and exit to panel (c) when the budget is spent.
arrow(f"M{BX + BW},589 H{BX + BW + 34} V160 H{BX + BW + 3}")
add(f'<text x="{BX + BW + 50}" y="380" font-size="12" fill="{INK2}" text-anchor="middle" '
    f'transform="rotate(90 {BX + BW + 50} 380)">repeat until B rows are labeled</text>')
arrow(f"M{BX + BW + 34},160 V128 H1048")
text(BX + BW + 44, 120, "B reached", size=12, fill=INK2)

# Shared labels: one labeled row updates its cluster in every encoder.
arrow(f"M{BX},610 H458 V486 H421", color=VAR_STROKE, dashed=True)
text(BX + 4, 658, "Dashed arrow = shared labels (option): each labeled row also", size=12, fill=VAR_STROKE)
text(BX + 4, 675, "updates its cluster in every other encoder (one label, M updates)", size=12, fill=VAR_STROKE)
text(BX + 4, 694, "Warm start (option): one query-seeded batch per encoder first", size=12, fill=INK2)

# ---------- Panel (c) ----------
CX, CW = 1050, 310
steps = [
    (["B labeled rows", "proxy-embedding vectors + oracle labels"], FIX_FILL, FIX_STROKE),
    (["Fit logistic regression", "class_weight = balanced"], FIX_FILL, FIX_STROKE),
    (["Pick the score cutoff on val", "best F1 on 15,929 val comments"], FIX_FILL, FIX_STROKE),
    (["Score every row in the table", "one dot product per row, no LLM calls"], FIX_FILL, FIX_STROKE),
    (["Report on held-out test", "precision · recall · F1 · AP + bootstrap CIs"], FIX_FILL, FIX_STROKE),
]
y = 108
for i, (lines, fill, stroke) in enumerate(steps):
    labeled_box(CX, y, CW, 56, lines, fill, stroke)
    if i < len(steps) - 1:
        arrow(f"M{CX + CW / 2},{y + 56} V{y + 78}")
    y += 80
for i, s in enumerate(["Every sampler feeds this same proxy,",
                       "so a better F1 means better training rows,",
                       "not a better classifier."]):
    text(CX + CW / 2, 548 + 19 * i, s, size=12.5, fill=INK, anchor="middle")
text(CX + CW / 2, 640, "Ceiling = this panel trained on", size=12, fill=INK2, anchor="middle")
text(CX + CW / 2, 658, "all pool labels (F1 ≈ 0.43 for threat)", size=12, fill=INK2, anchor="middle")

# ---------- Legend ----------
ly = 740
items = [("box", VAR_FILL, VAR_STROKE, "varies between methods (research variable)"),
         ("box", FIX_FILL, FIX_STROKE, "fixed for every method"),
         ("box", LLM_FILL, LLM_STROKE, "oracle / LLM labels (costly)"),
         ("dot", POS, None, "positive row"), ("dot", NEG, None, "negative row")]
x = 24
for kind, fill, stroke, label in items:
    if kind == "box":
        add(f'<rect x="{x}" y="{ly - 12}" width="22" height="15" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        x += 30
    else:
        add(f'<circle cx="{x + 6}" cy="{ly - 4.5}" r="4.5" fill="{fill}"/>')
        x += 18
    text(x, ly, label, size=12.5, fill=INK2)
    x += len(label) * 6.9 + 26
text(24, 770, "Schematic only: the cluster scatters are illustrative, not real data. Steps ①–⑤ are proposed; LTS-lite (② alone, one encoder) is implemented.",
     size=11.5, fill=MUTED, style="italic")

add("</svg>")
open(sys.argv[1], "w").write("\n".join(out))
print("wrote", sys.argv[1])
