"""Screening diagnostic: what share of pool positives fall inside the top-s fraction ranked by a cheap filter?

Retrospective (reads pool labels), so it is a diagnostic, never a sampler. Filters: TF-IDF and
MiniLM cosine to each predicate phrasing in run_baselines.py; random subsampling is the reference.

Usage: python -I scripts/filter_recall.py data/jigsaw
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from embedseek.data import load_split

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
FRACS = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50]

data_dir = Path(sys.argv[1])
pool = load_split(data_dir, "pool", "all-MiniLM-L6-v2")
vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=500_000)
X = vec.fit_transform(pool.df.comment_text.tolist())
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

rows = []
for label, phrasings in QUERIES.items():
    y = pool.y(label)
    for frac in FRACS:
        rows.append({"label": label, "filter": "random", "variant": "", "frac": frac, "share_reachable": frac})
    for variant, text in phrasings.items():
        scores = {
            "tfidf": (X @ vec.transform([text]).T).toarray().ravel(),
            "minilm": pool.emb @ encoder.encode(text, normalize_embeddings=True),
        }
        for name, s in scores.items():
            order = np.argsort(-s)
            for frac in FRACS:
                rows.append({"label": label, "filter": name, "variant": variant, "frac": frac,
                             "share_reachable": y[order[:int(frac * len(y))]].sum() / y.sum()})

out = Path("results") / "filter_recall_all-MiniLM-L6-v2.csv"
df = pd.DataFrame(rows)
df.to_csv(out, index=False)
print(df.pivot_table(index=["label", "filter", "variant"], columns="frac", values="share_reachable").round(2).to_string())
print(f"\nwrote {out}")
