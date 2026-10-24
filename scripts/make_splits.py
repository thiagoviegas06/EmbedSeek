"""Build frozen pool / val / test splits for Jigsaw.

- Deduplicates on normalized text (lowercase, collapsed whitespace).
- Drops pool/val rows whose text also appears in test (leakage).
- Test = official test rows with real labels (not -1).
- Val = stratified 10% of deduplicated train, stratified on (threat, identity_hate).

Usage: python -I scripts/make_splits.py data/jigsaw/raw data/jigsaw/splits
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 42
VAL_FRAC = 0.10


def norm(s: pd.Series) -> pd.Series:
    return s.str.lower().str.split().str.join(" ")


raw, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(raw / "train.csv")
test = pd.read_csv(raw / "test.csv").merge(pd.read_csv(raw / "test_labels.csv"), on="id")
test = test[(test[LABELS] != -1).all(axis=1)].reset_index(drop=True)

stats = {"train_raw": len(train), "test_scored": len(test)}

# Dedup within train; report how many duplicate groups had conflicting labels.
train["_norm"] = norm(train.comment_text)
conflicts = int((train.groupby("_norm")[LABELS].nunique() > 1).any(axis=1).sum())
train = train.drop_duplicates("_norm", keep="first")
stats["train_after_dedup"] = len(train)
stats["dup_groups_with_label_conflict"] = conflicts

# Dedup within test, then remove train rows that also occur in test.
test["_norm"] = norm(test.comment_text)
test = test.drop_duplicates("_norm", keep="first")
stats["test_after_dedup"] = len(test)
overlap = train._norm.isin(set(test._norm))
stats["train_rows_overlapping_test_removed"] = int(overlap.sum())
train = train[~overlap]

strat = train.threat * 2 + train.identity_hate
pool, val = train_test_split(train, test_size=VAL_FRAC, stratify=strat, random_state=SEED)

cols = ["id", "comment_text"] + LABELS
for name, df in [("pool", pool), ("val", val), ("test", test)]:
    df[cols].reset_index(drop=True).to_parquet(out / f"{name}.parquet", index=False)
    stats[name] = {"rows": len(df), **{l: int(df[l].sum()) for l in LABELS}}

(out / "split_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
