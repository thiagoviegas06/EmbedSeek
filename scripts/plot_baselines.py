"""Plot baseline results: rows = predicates, columns = positives found / test F1 / test AP.

Usage: python -I scripts/plot_baselines.py results/baselines_all-MiniLM-L6-v2.csv [more.csv ...] results/figures/baselines_minilm.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Reference palette (dataviz skill), light mode. Slots 1-4 validated (adjacent pairs) for CVD.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [  # (sampler, variant, label, color, marker)
    ("random", "", "Random (mean ± 1 sd, 10 seeds)", "#2a78d6", "o"),
    ("query", "description", "Query: predicate description", "#eb6834", "s"),
    ("query", "pseudo_example", "Query: comment-like pseudo-example", "#1baf7a", "^"),
    ("lts_lite", "k50", "LTS-lite: bandit over 50 clusters (mean ± 1 sd)", "#eda100", "D"),
]
PANELS = [("n_pos", "Positives found"), ("f1", "Test F1"), ("avg_precision", "Test average precision")]
TITLES = {"threat": "threat  (0.30% of pool)", "identity_hate": "identity_hate  (0.88% of pool)"}

df = pd.concat([pd.read_csv(f) for f in sys.argv[1:-1]]).fillna({"variant": ""})
out = Path(sys.argv[-1])
out.parent.mkdir(parents=True, exist_ok=True)
labels = list(dict.fromkeys(df.label))
budgets = sorted(df[df.sampler == "random"].budget.unique())

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK_2,
                     "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
fig, axes = plt.subplots(len(labels), len(PANELS), figsize=(13, 4.2 * len(labels)), facecolor=SURFACE)

for r, label in enumerate(labels):
    sub = df[df.label == label]
    ceiling = sub[sub.sampler == "ceiling"].iloc[0]
    for c, (metric, title) in enumerate(PANELS):
        ax = axes[r, c]
        ax.set_facecolor(SURFACE)
        for sampler, variant, name, color, marker in SERIES:
            s = sub[(sub.sampler == sampler) & (sub.variant == variant)]
            if s.empty:
                continue
            g = s.groupby("budget")[metric].agg(["mean", "std"]).reindex(budgets)
            ax.plot(budgets, g["mean"], color=color, lw=2, marker=marker, ms=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
            if s.seed.nunique() > 1:  # seeded sampler: show spread across seeds
                ax.fill_between(budgets, (g["mean"] - g["std"]).clip(lower=0), g["mean"] + g["std"],
                                color=color, alpha=0.12, lw=0, zorder=2)
        if metric != "n_pos":  # ceiling = proxy trained on every pool label
            ax.axhline(ceiling[metric], color=INK_2, lw=1.2, ls=(0, (4, 3)), zorder=1)
            ax.annotate(f"ceiling (all {int(ceiling.n_labeled):,} labels) {ceiling[metric]:.2f}",
                        xy=(budgets[0], ceiling[metric]), xytext=(0, 4), textcoords="offset points",
                        fontsize=8.5, color=INK_2, va="bottom")
            ax.set_ylim(0, max(0.55, ceiling[metric] + 0.1))
        else:
            ax.set_ylim(bottom=0)
        ax.set_xscale("log")
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.minorticks_off()
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if r == len(labels) - 1:
            ax.set_xlabel("Label budget B (LLM/oracle labels)")
        if r == 0:
            ax.set_title(title, fontsize=11, color=INK, loc="left", fontweight="bold")
        if c == 0:
            ax.set_ylabel(TITLES[label], fontsize=10.5, color=INK)

handles, names = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, names, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Baselines on Jigsaw: all-MiniLM-L6-v2 for search and proxy features", x=0.01, ha="left",
             y=1.045, fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
print(f"wrote {out}")
