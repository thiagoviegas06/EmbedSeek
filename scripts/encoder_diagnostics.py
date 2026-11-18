"""Encoder diagnostics (go/no-go for the encoder layer): do MiniLM, mpnet and bge concentrate *different* positives?

Per encoder x predicate:
- ceiling:  the proxy trained on all pool labels with that encoder's features (F1, AP, 95% bootstrap CIs)
- clusters: k-means (k=50, the cached clusters every script shares); share of positives in the top-5 clusters
- query:    top-1000 pool rows by cosine to the comment-like query from run_baselines.py; positives found
Across encoders:
- clusters: Jaccard of the positive sets each encoder's top-5 clusters hold; positives held by only one encoder
- query:    Jaccard of the top-1000 result sets (all rows, and positives only); positives found by any encoder

Retrospective: ranks clusters with true pool labels, so it is analysis only, never a sampler input.
bge gets its retrieval instruction on the query only (ranker "bge-base-en-v1.5"); "bge-base-en-v1.5_noprefix" is
the plain query, since the model card recommends no instruction for symmetric similarity (comment vs comment).

Writes results/encoder_diagnostics.csv in long form: label, kind, metric, model_a, model_b, value.

Usage: python -I scripts/encoder_diagnostics.py data/jigsaw
"""
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from embedseek.data import load_clusters, load_split
from embedseek.proxy import ci_columns, train_and_evaluate
from embedseek.samplers import query_seeded

MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}
BGE_PREFIX = "Represent this sentence for searching relevant passages: "   # bge-*-v1.5 model card
PREDICATES = ["threat", "identity_hate"]
QUERY = {   # the comment-like ("pseudo_example") phrasings from run_baselines.py
    "threat": "I am going to find you and kill you.",
    "identity_hate": "people of your race and religion are disgusting and should all be gone",
}
K, TOP_CLUSTERS, TOP_Q = 50, 5, 1000

data_dir = Path(sys.argv[1])
rows = []


def add(label, kind, metric, value, model_a="", model_b=""):
    rows.append({"label": label, "kind": kind, "metric": metric, "model_a": model_a, "model_b": model_b,
                 "value": float(value)})


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else float("nan")


cluster_pos = {}   # (label, model) -> pool rows of positives inside that encoder's top-5 clusters
query_top = {}     # (label, ranker) -> the ranker's top-1000 pool rows
y_pool = {}
for slug, name in MODELS.items():
    pool, val, test = (load_split(data_dir, s, slug) for s in ("pool", "val", "test"))
    t0 = time.perf_counter()
    cluster_ids = load_clusters(data_dir, slug, K)
    print(f"[{slug}] k-means (k={K}): {time.perf_counter() - t0:.1f}s", flush=True)

    encoder = SentenceTransformer(name, device="cpu")
    rankers = {slug: BGE_PREFIX, f"{slug}_noprefix": ""} if slug.startswith("bge") else {slug: ""}

    for label in PREDICATES:
        y, y_val, y_test = pool.y(label), val.y(label), test.y(label)
        y_pool[label] = y

        t0 = time.perf_counter()
        res = train_and_evaluate(pool.emb, y, val.emb, y_val, test.emb, y_test)
        for metric, v in {"f1": res.f1, "ap": res.avg_precision, **ci_columns(y_test, res)}.items():
            add(label, "ceiling", metric, v, slug)
        print(f"[{slug}] {label} ceiling F1 {res.f1:.3f} AP {res.avg_precision:.3f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

        per_cluster = np.bincount(cluster_ids[y == 1], minlength=K)
        in_top = np.isin(cluster_ids, np.argsort(-per_cluster)[:TOP_CLUSTERS])
        cluster_pos[label, slug] = set(np.flatnonzero(in_top & (y == 1)))
        add(label, "clusters", "top5_pos_share", len(cluster_pos[label, slug]) / y.sum(), slug)
        add(label, "clusters", "top5_row_share", in_top.mean(), slug)   # how much of the pool those clusters are

        for ranker, prefix in rankers.items():
            q = encoder.encode(prefix + QUERY[label], normalize_embeddings=True).astype(np.float32)
            query_top[label, ranker] = query_seeded(pool.emb, q, TOP_Q)
            add(label, "query", "positives_found", y[query_top[label, ranker]].sum(), ranker)

# ---- Overlap across encoders --------------------------------------------------------------------
slugs = list(MODELS)
for label in PREDICATES:
    y = y_pool[label]
    held = {m: cluster_pos[label, m] for m in slugs}
    for a, b in combinations(slugs, 2):
        add(label, "clusters", "pos_jaccard", jaccard(held[a], held[b]), a, b)
    for m in slugs:
        others = set().union(*(held[o] for o in slugs if o != m))
        add(label, "clusters", "pos_only_this_encoder", len(held[m] - others), m)
    add(label, "clusters", "pos_any_encoder", len(set().union(*held.values())))
    add(label, "clusters", "pos_all_encoders", len(set.intersection(*held.values())))
    add(label, "clusters", "pos_best_single", max(len(s) for s in held.values()))
    add(label, "clusters", "pos_total", y.sum())

    rankers = [r for (lbl, r) in query_top if lbl == label]
    top = {r: set(query_top[label, r]) for r in rankers}
    found = {r: {i for i in top[r] if y[i]} for r in rankers}
    for a, b in combinations(rankers, 2):
        add(label, "query", "top1000_jaccard", jaccard(top[a], top[b]), a, b)
        add(label, "query", "pos_jaccard", jaccard(found[a], found[b]), a, b)
    add(label, "query", "pos_any_encoder", len(set().union(*(found[m] for m in slugs))))   # prefixed bge only
    add(label, "query", "pos_best_single", max(len(found[m]) for m in slugs))

out = Path("results") / "encoder_diagnostics.csv"
df = pd.DataFrame(rows)
df.to_csv(out, index=False)

pd.set_option("display.width", 200)
for kind, g in df.groupby("kind", sort=False):
    print(f"\n== {kind}")
    print(g.pivot_table(index=["metric", "model_a", "model_b"], columns="label", values="value", sort=False)
           .round(3).to_string())
print(f"\nwrote {out}")
