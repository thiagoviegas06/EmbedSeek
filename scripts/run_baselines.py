"""Ceiling, random floor, and query-seeded baselines for one embedding model.

- ceiling: proxy trained on every pool label (best case for this proxy + embedding)
- random:  uniform samples at each budget x 10 seeds
- query:   top-B pool items by similarity to a predicate phrasing (deterministic)

Usage: python -I scripts/run_baselines.py data/jigsaw all-MiniLM-L6-v2 sentence-transformers/all-MiniLM-L6-v2
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from embedseek.data import load_split
from embedseek.proxy import train_and_evaluate
from embedseek.samplers import query_seeded, random_sample

PREDICATES = ["threat", "identity_hate"]
BUDGETS = [100, 250, 500, 1000]
SEEDS = range(10)
QUERIES = {
    "threat": {
        "description": "a threat of violence or harm against someone",
        "pseudo_example": "I am going to find you and kill you.",
    },
    "identity_hate": {
        "description": "hateful insults attacking a person's race, religion, ethnicity, gender or sexual orientation",
        "pseudo_example": "people of your race and religion are disgusting and should all be gone",
    },
}

data_dir, model_slug, model_name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
pool, val, test = (load_split(data_dir, s, model_slug) for s in ("pool", "val", "test"))
encoder = SentenceTransformer(model_name, device="cpu")

rows = []
for label in PREDICATES:
    y_pool, y_val, y_test = pool.y(label), val.y(label), test.y(label)

    def evaluate(idx, sampler, budget, seed, variant=""):
        t0 = time.perf_counter()
        res = train_and_evaluate(pool.emb[idx], y_pool[idx], val.emb, y_val, test.emb, y_test)
        rows.append({"model": model_slug, "label": label, "sampler": sampler, "variant": variant,
                     "budget": budget, "seed": seed, **res.as_dict(),
                     "train_eval_secs": round(time.perf_counter() - t0, 2)})

    evaluate(np.arange(len(y_pool)), "ceiling", len(y_pool), -1)
    print(f"[{label}] ceiling done", flush=True)

    for budget in BUDGETS:
        for seed in SEEDS:
            evaluate(random_sample(len(y_pool), budget, seed), "random", budget, seed)
    print(f"[{label}] random done", flush=True)

    for variant, text in QUERIES[label].items():
        q = encoder.encode(text, normalize_embeddings=True).astype(np.float32)
        for budget in BUDGETS:
            evaluate(query_seeded(pool.emb, q, budget), "query", budget, -1, variant)
    print(f"[{label}] query done", flush=True)

out = Path("results") / f"baselines_{model_slug}.csv"
df = pd.DataFrame(rows)
df.to_csv(out, index=False)

summary = (df.groupby(["label", "sampler", "variant", "budget"], sort=False)
             .agg(n_pos=("n_pos", "mean"), trained=("trained", "mean"),
                  f1=("f1", "mean"), f1_std=("f1", "std"), ap=("avg_precision", "mean"))
             .round(3))
print(summary.to_string())
print(f"\nwrote {out}")
