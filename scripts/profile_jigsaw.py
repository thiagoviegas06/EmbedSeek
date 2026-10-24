"""Profile the Jigsaw Toxic Comment data: sizes, label prevalence, overlap, lengths."""
import sys
import pandas as pd

raw = sys.argv[1]
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train = pd.read_csv(f"{raw}/train.csv")
test = pd.read_csv(f"{raw}/test.csv")
tl = pd.read_csv(f"{raw}/test_labels.csv")

print(f"train rows: {len(train):,}  test rows: {len(test):,}")
scored = tl[(tl[labels] != -1).all(axis=1)]
print(f"test rows with real labels (not -1): {len(scored):,}")
print(f"null comments train: {train.comment_text.isna().sum()}, dup ids: {train.id.duplicated().sum()}")

print("\nPrevalence in train")
for l in labels:
    n = int(train[l].sum())
    print(f"  {l:13s} positives={n:6,d}  prevalence={n/len(train):.4%}")
print(f"  any label     positives={int((train[labels].sum(axis=1) > 0).sum()):,}")
print(f"  clean (none)  {int((train[labels].sum(axis=1) == 0).sum()):,}")

print("\nPrevalence in scored test")
for l in labels:
    n = int((scored[l] == 1).sum())
    print(f"  {l:13s} positives={n:6,d}  prevalence={n/len(scored):.4%}")

print("\nLabel co-occurrence among threat positives (train)")
th = train[train.threat == 1]
for l in labels:
    print(f"  {l:13s} {th[l].mean():.1%}")

print("\nComment length (chars) quantiles")
print(train.comment_text.str.len().describe(percentiles=[.5, .9, .99]).round(0).to_string())

print("\nSample threat positives")
for t in th.comment_text.sample(5, random_state=0):
    print("  -", t[:200].replace("\n", " "))
