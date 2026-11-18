"""Two-level bandit (encoder -> cluster) vs its baselines and controls, with fixed proxy embeddings.

Search runs over MiniLM, mpnet and bge (cached k-means clusters per encoder). Sampling never sees the proxy,
so each sample is drawn once and then evaluated with every <proxy-embedding> given.
Rows written (sampler / variant):
- random / -:                      floor
- query / <encoder>:               top-B by the comment-like query in that encoder (deterministic)
- multi_query / -:                 the encoders take turns adding their top query rows (deterministic)
- warm_random / -:                 the bandit's warm start (75 query rows), then random: does the bandit add anything?
- lts_lite / <encoder>:            LTS-lite searching that encoder alone
- lts_lite / <encoder>+warm:       the same after 75 query rows from that encoder: the bar to beat
- hier_lts / {share,noshare}+{cold,warm}:  the two-level bandit and its ablations
- hier_lts / share+{cold,warm}+rate:        the same, encoder level rewarded by positives / batch
- flat_lts / share+{cold,warm}:    every (encoder, cluster) pair as one arm: is the hierarchy needed?
The query is the comment-like one from run_baselines.py, embedded by each encoder (no bge instruction: docs/10).
`encoder_rounds` counts the rounds each encoder proposed, in the order MiniLM|mpnet|bge.

Writes results/bandit_<proxy>.csv (one row per run) and results/bandit_pairs_<proxy>.csv: for every pair of
methods, the difference of seed-mean F1 and AP with a 95% CI over test rows and sampler seeds (mean_diff_ci).

Usage: python -I scripts/run_bandit.py data/jigsaw bge-base-en-v1.5 [more proxy embeddings...]
"""
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from embedseek.data import load_clusters, load_split
from embedseek.proxy import _ci, bootstrap_samples, mean_diff_ci, train_and_evaluate
from embedseek.samplers import hierarchical_lts, lts_lite, multi_query, query_seeded, random_sample

MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}
PREDICATES = ["threat", "identity_hate"]
QUERY = {   # the comment-like ("pseudo_example") phrasings from run_baselines.py
    "threat": "I am going to find you and kill you.",
    "identity_hate": "people of your race and religion are disgusting and should all be gone",
}
BUDGETS = [100, 250, 500, 1000]
SEEDS = range(10)
K, BATCH = 50, 25
BANDITS = {   # (sampler, variant) -> hierarchical_lts options
    ("hier_lts", "share+cold"): dict(share_labels=True, warm=False),
    ("hier_lts", "noshare+cold"): dict(share_labels=False, warm=False),
    ("hier_lts", "share+warm"): dict(share_labels=True, warm=True),
    ("hier_lts", "noshare+warm"): dict(share_labels=False, warm=True),
    ("hier_lts", "share+cold+rate"): dict(share_labels=True, warm=False, encoder_reward="rate"),
    ("hier_lts", "share+warm+rate"): dict(share_labels=True, warm=True, encoder_reward="rate"),
    ("flat_lts", "share+cold"): dict(share_labels=True, warm=False, flat=True),
    ("flat_lts", "share+warm"): dict(share_labels=True, warm=True, flat=True),
}

data_dir, proxy_slugs = Path(sys.argv[1]), sys.argv[2:]
slugs = list(MODELS)
WARM_ROWS = BATCH * len(slugs)   # query rows the bandit's warm start labels; the controls get the same number

# Search spaces: each encoder's pool embedding, its clusters, and its embedding of the queries.
splits = {slug: {s: load_split(data_dir, s, slug) for s in ("pool", "val", "test")}
          for slug in dict.fromkeys(slugs + proxy_slugs)}
pool = splits[slugs[0]]["pool"]   # labels and row order are the same for every encoder
search_embs = [splits[slug]["pool"].emb for slug in slugs]
cluster_ids = [load_clusters(data_dir, slug, K) for slug in slugs]
queries = {label: [] for label in PREDICATES}
for name in MODELS.values():
    encoder = SentenceTransformer(name, device="cpu")
    for label in PREDICATES:
        queries[label].append(encoder.encode(QUERY[label], normalize_embeddings=True).astype(np.float32))

rows = {p: [] for p in proxy_slugs}
pairs = {p: [] for p in proxy_slugs}
for label in PREDICATES:
    y_pool = pool.y(label)
    oracle = lambda i: y_pool[i]
    q = queries[label]

    for budget in BUDGETS:
        boot = {p: {} for p in proxy_slugs}   # proxy -> (sampler, variant) -> per-seed bootstrap (f1s, aps)

        def record(sampler, variant, seed, idx, sample_secs, trace=None):
            per_encoder = np.bincount([t[0] for t in trace], minlength=len(slugs)) if trace else []
            for p in proxy_slugs:
                sp = splits[p]
                y_test = sp["test"].y(label)
                res = train_and_evaluate(sp["pool"].emb[idx], y_pool[idx], sp["val"].emb, sp["val"].y(label),
                                         sp["test"].emb, y_test)
                f1s, aps = bootstrap_samples(y_test, res.test_scores, res.threshold)   # same resamples every run
                boot[p].setdefault((sampler, variant), []).append((f1s, aps))
                (f1_lo, f1_hi), (ap_lo, ap_hi) = _ci(f1s), _ci(aps)
                rows[p].append({"model": p, "label": label, "sampler": sampler, "variant": variant,
                                "budget": budget, "seed": seed, **res.as_dict(), "sample_secs": round(sample_secs, 2),
                                "f1_lo": f1_lo, "f1_hi": f1_hi, "ap_lo": ap_lo, "ap_hi": ap_hi,   # 95% test-set CI
                                "encoder_rounds": "|".join(map(str, per_encoder))})

        def timed(sampler, variant, seed, draw, trace=None):
            t0 = time.perf_counter()
            idx = draw()
            record(sampler, variant, seed, idx, time.perf_counter() - t0, trace)

        # Deterministic: one run each.
        for m, slug in enumerate(slugs):
            timed("query", slug, 0, lambda: query_seeded(search_embs[m], q[m], budget))
        timed("multi_query", "-", 0, lambda: multi_query(search_embs, q, budget, BATCH))
        warm_rows = multi_query(search_embs, q, min(WARM_ROWS, budget), BATCH)

        for seed in SEEDS:
            timed("random", "-", seed, lambda: random_sample(len(y_pool), budget, seed))
            rest = np.setdiff1d(np.arange(len(y_pool)), warm_rows)
            timed("warm_random", "-", seed, lambda: np.concatenate([
                warm_rows, np.random.default_rng(seed).choice(rest, budget - len(warm_rows), replace=False)]))
            for m, slug in enumerate(slugs):
                timed("lts_lite", slug, seed, lambda: lts_lite(search_embs[m], cluster_ids[m], oracle, budget, seed))
                timed("lts_lite", f"{slug}+warm", seed, lambda: hierarchical_lts(
                    [search_embs[m]], [cluster_ids[m]], oracle, budget, seed, warm_start=[q[m]], warm_batch=WARM_ROWS))
            for (sampler, variant), opts in BANDITS.items():
                opts = dict(opts)
                warm_start = q if opts.pop("warm") else None
                trace = []
                timed(sampler, variant, seed, lambda: hierarchical_lts(
                    search_embs, cluster_ids, oracle, budget, seed, warm_start=warm_start, trace=trace, **opts), trace)

        # Every pair of methods: difference of seed-mean F1 / AP, CI over test rows and sampler seeds.
        for p in proxy_slugs:
            f1 = {k: np.array([r[0] for r in v]) for k, v in boot[p].items()}
            ap = {k: np.array([r[1] for r in v]) for k, v in boot[p].items()}
            point = pd.DataFrame(rows[p]).query("label == @label and budget == @budget") \
                      .groupby(["sampler", "variant"])[["f1", "avg_precision"]].mean()
            for a, b in combinations(f1, 2):
                (f1_lo, f1_hi), (ap_lo, ap_hi) = mean_diff_ci(f1[a], f1[b]), mean_diff_ci(ap[a], ap[b])
                pairs[p].append({"model": p, "label": label, "budget": budget,
                                 "method_a": "/".join(a), "method_b": "/".join(b),
                                 "f1_diff": point.loc[a, "f1"] - point.loc[b, "f1"], "f1_diff_lo": f1_lo, "f1_diff_hi": f1_hi,
                                 "ap_diff": point.loc[a, "avg_precision"] - point.loc[b, "avg_precision"],
                                 "ap_diff_lo": ap_lo, "ap_diff_hi": ap_hi})
        print(f"[{label}] B={budget} done", flush=True)

pd.set_option("display.width", 200)
for p in proxy_slugs:
    out, out_pairs = Path("results") / f"bandit_{p}.csv", Path("results") / f"bandit_pairs_{p}.csv"
    df = pd.DataFrame(rows[p])
    df.to_csv(out, index=False)
    pd.DataFrame(pairs[p]).to_csv(out_pairs, index=False)
    summary = (df.groupby(["label", "budget", "sampler", "variant"], sort=False)
                 .agg(n_pos=("n_pos", "mean"), n_pos_std=("n_pos", "std"), trained=("trained", "mean"),
                      f1=("f1", "mean"), f1_std=("f1", "std"), f1_lo=("f1_lo", "mean"), f1_hi=("f1_hi", "mean"),
                      ap=("avg_precision", "mean"), sample_secs=("sample_secs", "mean"))
                 .round(3))
    print(f"\n== proxy {p}")
    print(summary.to_string())
    print(f"wrote {out} and {out_pairs}")
