"""Varied warm start: several example sentences per predicate vs one, across one or three encoders (docs/11 §3.1).

Every warm start labels the same number of rows (75) before the follower takes over. Warm sources:
- 1s-1enc: the original sentence, the follower's encoder only (75 rows; = lts_lite/<encoder>+warm in run_bandit.py)
- 1s-3enc: the original sentence, MiniLM / mpnet / bge taking turns (25 rows each; the bandit's warm start)
- 5s-1enc: five sentences, the follower's encoder (15 rows each)
- 5s-3enc: five sentences x three encoders (5 rows per pair)
Followers: LTS-lite in bge (lts_bge), LTS-lite in MiniLM (lts_minilm), or random rows (random; its 1enc = bge).
Query-only controls (deterministic, no follower): query/1s-3enc and query/5s-3enc fill the whole budget by query.

Extra columns: warm_pos = positives among the warm rows; pos_spread = mean pairwise cosine distance between the
positives found, in the bge embedding (higher = more varied positives; NaN with fewer than 2).

Writes results/warm_start_<proxy>.csv and results/warm_start_pairs_<proxy>.csv (same columns as run_bandit.py).

Usage: python -I scripts/run_warm_start.py data/jigsaw bge-base-en-v1.5 [more proxy embeddings...]
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
from embedseek.samplers import hierarchical_lts, multi_query

MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}
PREDICATES = ["threat", "identity_hate"]
# Hand-written, comment-like examples of different kinds of each predicate. The first is the original
# pseudo_example from run_baselines.py, so the 1s variants reproduce earlier runs.
SENTENCES = {
    "threat": [
        "I am going to find you and kill you.",                                       # direct
        "You better watch your back, something bad is going to happen to you.",       # veiled
        "I know where you live and I will burn your house down.",                     # property
        "If you delete my edit again I will come after you and your family.",          # conditional
        "Keep talking and I will beat you until you can't walk.",                      # assault
    ],
    "identity_hate": [
        "people of your race and religion are disgusting and should all be gone",     # race / religion
        "gay people are disgusting perverts and should be kept away from children",    # sexual orientation
        "go back to your own country, immigrants like you are ruining everything",     # nationality
        "all muslims are terrorists and should be banned from this country",           # religion
        "women are too stupid to edit articles, stick to the kitchen",                 # gender
    ],
}
BUDGETS = [100, 250, 500, 1000]
SEEDS = range(10)
K, WARM = 50, 75
FOLLOWERS = {"lts_bge": "bge-base-en-v1.5", "lts_minilm": "all-MiniLM-L6-v2", "random": "bge-base-en-v1.5"}
SPREAD_SPACE = "bge-base-en-v1.5"

data_dir, proxy_slugs = Path(sys.argv[1]), sys.argv[2:]
slugs = list(MODELS)
assert all(len(v) * len(slugs) * (WARM // (len(v) * len(slugs))) == WARM for v in SENTENCES.values())

splits = {slug: {s: load_split(data_dir, s, slug) for s in ("pool", "val", "test")}
          for slug in dict.fromkeys(slugs + proxy_slugs)}
pool = splits[slugs[0]]["pool"]   # labels and row order are the same for every encoder
emb = {slug: splits[slug]["pool"].emb for slug in slugs}
clusters = {slug: load_clusters(data_dir, slug, K) for slug in slugs}
qemb = {}   # (label, encoder) -> one normalized embedding per sentence; no bge instruction (docs/10)
for slug, name in MODELS.items():
    encoder = SentenceTransformer(name, device="cpu")
    for label in PREDICATES:
        qemb[label, slug] = list(encoder.encode(SENTENCES[label], normalize_embeddings=True).astype(np.float32))


def warm_rows(label, source, enc, n):
    """The `n` warm rows for a source (1s/5s x 1enc/3enc): (encoder, sentence) pairs take turns."""
    n_sent = 1 if source.startswith("1s") else len(SENTENCES[label])
    encs = [enc] if source.endswith("1enc") else slugs
    pairs = [(e, s) for s in range(n_sent) for e in encs]
    return multi_query([emb[e] for e, _ in pairs], [qemb[label, e][s] for e, s in pairs], n, WARM // len(pairs))


def spread(idx, y):
    pos = idx[y[idx] == 1]
    if len(pos) < 2:
        return np.nan
    e = emb[SPREAD_SPACE][pos]
    sim = e @ e.T
    return float(1 - (sim.sum() - len(pos)) / (len(pos) * (len(pos) - 1)))


rows = {p: [] for p in proxy_slugs}
pairs = {p: [] for p in proxy_slugs}
for label in PREDICATES:
    y_pool = pool.y(label)
    oracle = lambda i: y_pool[i]

    for budget in BUDGETS:
        boot = {p: {} for p in proxy_slugs}

        def record(sampler, variant, seed, idx, sample_secs, warm):
            extra = {"warm_pos": int(y_pool[warm].sum()), "pos_spread": spread(idx, y_pool)}
            for p in proxy_slugs:
                sp = splits[p]
                y_test = sp["test"].y(label)
                res = train_and_evaluate(sp["pool"].emb[idx], y_pool[idx], sp["val"].emb, sp["val"].y(label),
                                         sp["test"].emb, y_test)
                f1s, aps = bootstrap_samples(y_test, res.test_scores, res.threshold)
                boot[p].setdefault((sampler, variant), []).append((f1s, aps))
                (f1_lo, f1_hi), (ap_lo, ap_hi) = _ci(f1s), _ci(aps)
                rows[p].append({"model": p, "label": label, "sampler": sampler, "variant": variant,
                                "budget": budget, "seed": seed, **res.as_dict(), "sample_secs": round(sample_secs, 2),
                                "f1_lo": f1_lo, "f1_hi": f1_hi, "ap_lo": ap_lo, "ap_hi": ap_hi, **extra})

        # Query-only controls: the whole budget by query, no follower.
        for source in ("1s-3enc", "5s-3enc"):
            t0 = time.perf_counter()
            idx = warm_rows(label, source, None, budget)
            record("query", source, 0, idx, time.perf_counter() - t0, idx[:WARM])

        for follower, enc in FOLLOWERS.items():
            for source in ("1s-1enc", "1s-3enc", "5s-1enc", "5s-3enc"):
                warm = warm_rows(label, source, enc, min(WARM, budget))
                for seed in SEEDS:
                    t0 = time.perf_counter()
                    if follower == "random":
                        rest = np.setdiff1d(np.arange(len(y_pool)), warm)
                        idx = np.concatenate([warm, np.random.default_rng(seed).choice(
                            rest, budget - len(warm), replace=False)])
                    else:
                        idx = hierarchical_lts([emb[enc]], [clusters[enc]], oracle, budget, seed, warm_rows=warm)
                    record(follower, source, seed, idx, time.perf_counter() - t0, warm)

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
    out, out_pairs = Path("results") / f"warm_start_{p}.csv", Path("results") / f"warm_start_pairs_{p}.csv"
    df = pd.DataFrame(rows[p])
    df.to_csv(out, index=False)
    pd.DataFrame(pairs[p]).to_csv(out_pairs, index=False)
    summary = (df.groupby(["label", "budget", "sampler", "variant"], sort=False)
                 .agg(warm_pos=("warm_pos", "mean"), n_pos=("n_pos", "mean"), spread=("pos_spread", "mean"),
                      f1=("f1", "mean"), f1_lo=("f1_lo", "mean"), f1_hi=("f1_hi", "mean"),
                      ap=("avg_precision", "mean"))
                 .round(3))
    print(f"\n== proxy {p}")
    print(summary.to_string())
    print(f"wrote {out} and {out_pairs}")
