"""Competing baseline: LTS-lite (Thompson sampling over k-means clusters of one embedding).

Usage: python -I scripts/run_lts_lite.py data/jigsaw all-MiniLM-L6-v2 [n_clusters]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from embedseek.data import load_split
from embedseek.proxy import train_and_evaluate
from embedseek.samplers import lts_lite

PREDICATES = ["threat", "identity_hate"]
BUDGETS = [100, 250, 500, 1000]
SEEDS = range(10)

data_dir, model_slug = Path(sys.argv[1]), sys.argv[2]
n_clusters = int(sys.argv[3]) if len(sys.argv) > 3 else 50
pool, val, test = (load_split(data_dir, s, model_slug) for s in ("pool", "val", "test"))

t0 = time.perf_counter()
cluster_ids = KMeans(n_clusters=n_clusters, n_init=1, random_state=0).fit_predict(pool.emb)
cluster_secs = time.perf_counter() - t0
print(f"k-means (k={n_clusters}) on {len(pool.emb):,} rows: {cluster_secs:.1f}s", flush=True)

rows = []
for label in PREDICATES:
    y_pool, y_val, y_test = pool.y(label), val.y(label), test.y(label)

    # Retrospective diagnostic (never visible to the sampler): how concentrated are positives?
    per_cluster = np.bincount(cluster_ids[y_pool == 1], minlength=n_clusters)
    top5 = np.sort(per_cluster)[::-1][:5].sum() / per_cluster.sum()
    print(f"[{label}] {per_cluster.sum()} positives; top-5 of {n_clusters} clusters hold {top5:.0%}", flush=True)

    for budget in BUDGETS:
        for seed in SEEDS:
            t0 = time.perf_counter()
            idx = lts_lite(pool.emb, cluster_ids, lambda i: y_pool[i], budget, seed)
            sample_secs = time.perf_counter() - t0
            res = train_and_evaluate(pool.emb[idx], y_pool[idx], val.emb, y_val, test.emb, y_test)
            rows.append({"model": model_slug, "label": label, "sampler": "lts_lite",
                         "variant": f"k{n_clusters}", "budget": budget, "seed": seed, **res.as_dict(),
                         "sample_secs": round(sample_secs, 2), "cluster_secs": round(cluster_secs, 1)})
        print(f"[{label}] B={budget} done", flush=True)

out = Path("results") / f"lts_lite_{model_slug}.csv"
df = pd.DataFrame(rows)
df.to_csv(out, index=False)
summary = (df.groupby(["label", "budget"])
             .agg(n_pos=("n_pos", "mean"), n_pos_std=("n_pos", "std"), trained=("trained", "mean"),
                  f1=("f1", "mean"), f1_std=("f1", "std"), ap=("avg_precision", "mean"),
                  sample_secs=("sample_secs", "mean"))
             .round(3))
print(summary.to_string())
print(f"\nwrote {out}")
