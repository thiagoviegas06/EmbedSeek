"""Figures for docs/08_metrics_reference.md.

1. metrics_precision_recall.png: what precision / recall / F1 count, on the threat ceiling run
2. metrics_found_vs_f1.png:      positives found (sampling) vs test F1 (evaluation), per sampler
3. metrics_screening.png:        share of pool positives reachable after screening to the top-s fraction

Usage: python -I scripts/plot_metrics_reference.py results/figures
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Reference palette (dataviz skill), light mode; same sampler -> color mapping as plot_baselines.py.
SURFACE, INK, INK_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TITLES = {"threat": "threat  (0.30% of pool)", "identity_hate": "identity_hate  (0.88% of pool)"}

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK_2,
                     "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


base = pd.read_csv("results/baselines_all-MiniLM-L6-v2.csv").fillna({"variant": ""})
lts = pd.read_csv("results/lts_lite_all-MiniLM-L6-v2.csv").fillna({"variant": ""})
runs = pd.concat([base, lts])

# --- 1. Precision / recall as two bars over the same true positives ---------------------------
ceil = base[(base.label == "threat") & (base.sampler == "ceiling")].iloc[0]
n_real = 211  # threat positives in the test split (split_stats.json)
tp = round(ceil.recall * n_real)
flagged = round(tp / ceil.precision)
missed, false_alarms = n_real - tp, flagged - tp

fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=SURFACE)
style(ax)
ax.grid(False)
bars = [  # (row label, [(width, color, text)])
    (f"Real threats in test\n({n_real})", [(tp, BLUE, f"caught: {tp}"), (missed, MUTED, f"missed: {missed}")]),
    (f"Flagged by the proxy\n({flagged})", [(tp, BLUE, f"real threats: {tp}"),
                                            (false_alarms, ORANGE, f"false alarms: {false_alarms}")]),
]
for y, (name, segments) in zip([1, 0], bars):
    left = 0
    for width, color, text in segments:
        ax.barh(y, width - 2, left=left, color=color, height=0.55, edgecolor=SURFACE, linewidth=0)
        ax.text(left + width / 2, y, text, ha="center", va="center", fontsize=10,
                color="white" if color != MUTED else INK)
        left += width
ax.set_yticks([1, 0], [b[0] for b in bars], fontsize=10, color=INK)
ax.set_xlim(0, max(n_real, flagged) + 5)
ax.set_xlabel("Number of test comments")
ax.text(n_real + 3, 1, f"recall = {tp}/{n_real} = {tp / n_real:.2f}", va="center", fontsize=10, color=INK)
ax.text(flagged + 3, 0, f"precision = {tp}/{flagged} = {tp / flagged:.2f}", va="center", fontsize=10, color=INK)
ax.set_xlim(0, flagged + 95)
ax.set_title(f"How F1 is computed: threat proxy trained on all pool labels (F1 = {ceil.f1:.2f})",
             loc="left", fontsize=12, fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(out_dir / "metrics_precision_recall.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")

# --- 2. Positives found (x) vs test F1 (y): related, not proportional --------------------------
SERIES = [
    ("random", "", "Random", BLUE, "o"),
    ("query", "description", "Query: description", ORANGE, "s"),
    ("query", "pseudo_example", "Query: comment-like", AQUA, "^"),
    ("lts_lite", "k50", "LTS-lite", YELLOW, "D"),
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
for ax, label in zip(axes, TITLES):
    style(ax)
    sub = runs[runs.label == label]
    for sampler, variant, name, color, marker in SERIES:
        g = sub[(sub.sampler == sampler) & (sub.variant == variant)].groupby("budget")[["n_pos", "f1"]].mean()
        ax.plot(g.n_pos.clip(lower=0.1), g.f1, color=color, lw=2, marker=marker, ms=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
        ax.annotate(f"B={g.index[-1]}", (g.n_pos.iloc[-1], g.f1.iloc[-1]), xytext=(0, 9),
                    textcoords="offset points", fontsize=8.5, color=INK_2, ha="center")
    c = sub[sub.sampler == "ceiling"].iloc[0]
    ax.axhline(c.f1, color=INK_2, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"ceiling (all pool labels) {c.f1:.2f}", xy=(0.01, c.f1), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_xscale("log")
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("Positives found in the labeled sample (log scale; mean over seeds)")
    ax.set_title(TITLES[label], loc="left", fontsize=11, fontweight="bold", color=INK)
axes[0].set_ylabel("Test F1 of the proxy")
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Positives found (sampling) vs test F1 (evaluation): each line goes B = 100 → 1000",
             x=0.01, ha="left", y=1.06, fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out_dir / "metrics_found_vs_f1.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")

# --- 3. Screening: share of positives reachable in the top-s fraction --------------------------
screen = pd.read_csv("results/filter_recall_all-MiniLM-L6-v2.csv").fillna({"variant": ""})
LINES = [  # color = query phrasing (as in figure 2), line style = filter; order fills legend columns
    ("minilm", "pseudo_example", "MiniLM, comment-like query", AQUA, "-", "^"),
    ("tfidf", "pseudo_example", "TF-IDF, comment-like query", AQUA, (0, (4, 2)), "^"),
    ("minilm", "description", "MiniLM, description query", ORANGE, "-", "s"),
    ("tfidf", "description", "TF-IDF, description query", ORANGE, (0, (4, 2)), "s"),
]
NOTE_AT = {"threat": (26, 0.45), "identity_hate": (13, 0.92)}  # annotation text positions in empty space
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
for ax, label in zip(axes, TITLES):
    style(ax)
    sub = screen[screen.label == label]
    r = sub[sub["filter"] == "random"]
    for filt, variant, name, color, ls, marker in LINES:
        s = sub[(sub["filter"] == filt) & (sub.variant == variant)]
        ax.plot(s.frac * 100, s.share_reachable, color=color, lw=2, ls=ls, marker=marker, ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
    ax.plot(r.frac * 100, r.share_reachable, color=INK_2, lw=1.2, ls=(0, (1, 2)),
            label="Random subsample (reference)", zorder=2)
    at10 = sub[(sub["filter"] == "minilm") & (sub.variant == "pseudo_example") & (sub.frac == 0.10)].share_reachable.iloc[0]
    ax.annotate(f"{at10:.0%} of positives kept\nby embedding only 10%", xy=(10, at10), xytext=NOTE_AT[label],
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    ax.set_xlim(0, 52)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Fraction of pool rows passed to the expensive encoders (%)")
    ax.set_title(TITLES[label], loc="left", fontsize=11, fontweight="bold", color=INK)
axes[0].set_ylabel("Share of pool positives still reachable")
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Screening: a cheap filter keeps far more positives than a random subsample",
             x=0.01, ha="left", y=1.1, fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(out_dir / "metrics_screening.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
print(f"wrote 3 figures to {out_dir}")
