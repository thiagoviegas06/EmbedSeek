# Jigsaw Toxic Comments: first look

**Source:** HF mirror of the Kaggle "Toxic Comment Classification Challenge" (Wikipedia talk-page comments). Raw files in `data/jigsaw/raw/` (`train.csv` 160k rows, `test.csv`, `test_labels.csv`). Profiling script: [scripts/profile_jigsaw.py](../scripts/profile_jigsaw.py); full output: [data/jigsaw/profile_output.txt](../data/jigsaw/profile_output.txt).

Re-run: `python3 -I scripts/profile_jigsaw.py data/jigsaw/raw`

## Key facts
- Train: 159,571 comments, no nulls or duplicate ids. Test: 153,164 rows but **only 63,978 have real labels**; the rest are `-1` (unscored by Kaggle). Use only scored rows.
- Six binary labels, multi-label.

| label | train positives | train prev. | scored-test positives | test prev. |
|---|---|---|---|---|
| toxic | 15,294 | 9.58% | 6,090 | 9.52% |
| severe_toxic | 1,595 | 1.00% | 367 | 0.57% |
| obscene | 8,449 | 5.29% | 3,691 | 5.77% |
| **threat** | **478** | **0.30%** | **211** | **0.33%** |
| insult | 7,877 | 4.94% | 3,427 | 5.36% |
| **identity_hate** | **1,405** | **0.88%** | **712** | **1.11%** |

- Comment length: median 205 chars, p90 889, p99 3,444, max 5,000. Long tail; encoders truncate (MiniLM at 256 tokens), so pick a max length deliberately.
- 94% of `threat` positives are also `toxic`; 63% obscene, 64% insult, 20% identity_hate. So "threat" neighbors will be polluted by generic toxicity, which makes this a real test for query-based and anchor-based sampling.

## What this means for the project
- `threat` at 0.30% is a natural needle-in-a-haystack: a 500-label uniform sample yields ~1.5 positives. 1,000 labels → ~3. Good motivating example.
- 0.1% prevalence isn't available naturally, so we must subsample positives to build it (after fixing splits). 478 train positives is enough to build pools of several sizes.
- **Label noise:** a random threat sample included "Please stop... you will be blocked", which is a warning, not a threat. Human labels are imperfect, and our "oracle" inherits that. The LLM comparison later may disagree with these labels for good reason. Note it in the report.
- Subtype diversity: threats vary (violence, death wishes, doxxing/legal, ethnic) but no subtype labels exist. Co-labels (identity_hate, obscene) are a weak independent diversity proxy.

## Open items
- [ ] Decide split scheme: use official train for pool+val, scored test for test (keeps the human-labeled test disjoint).
- [ ] Check for near-duplicate comments (copy-paste spam inflates positives and distorts diversity).
- [ ] Benchmark embedding 160k comments with MiniLM on this machine.
