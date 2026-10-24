# Baseline results: MiniLM, Jigsaw (2026-10-24)

**Setup:** frozen splits (pool 143,354 / val 15,929 / test 63,939; see `data/jigsaw/splits/split_stats.json`). Embedding `all-MiniLM-L6-v2` serves as both the search space and the proxy features. Proxy = LR (`class_weight="balanced"`), threshold = best F1 on val, metrics on test.
**Reproduce:** `.venv/bin/python -I scripts/run_baselines.py data/jigsaw all-MiniLM-L6-v2 sentence-transformers/all-MiniLM-L6-v2` → `results/baselines_all-MiniLM-L6-v2.csv`; `scripts/run_lts_lite.py data/jigsaw all-MiniLM-L6-v2` → `results/lts_lite_all-MiniLM-L6-v2.csv`; plot with `scripts/plot_baselines.py <csv...> results/figures/baselines_minilm.png`

**Embedding cost:** 4.3 min for 223K comments on Apple MPS (pool 152 s, val 18 s, test 87 s). 6.9% of comments exceed 256 tokens; threat/identity_hate positives are truncated at about the same rate (4–7%).

![Baselines](../results/figures/baselines_minilm.png)

Figure: `scripts/plot_baselines.py`. Random and LTS-lite are means of 10 seeds with ±1 sd bands; query runs are deterministic; the dashed line is the full-label ceiling.

## Results (test F1 / average precision; random = mean of 10 seeds)

### threat (pool 0.30%, 427 positives; test 211 positives)
| Sampler | B=100 | 250 | 500 | 1000 |
|---|---|---|---|---|
| random: positives found | 0.2 | 1.2 | 1.4 | 4.3 |
| random: share of runs that could train | 20% | 70% | 70% | 100% |
| random: F1 | 0.03 | 0.09 | 0.09 | 0.21 |
| query "description": positives | 3 | 4 | 6 | 8 |
| query "description": F1 | 0.28 | 0.27 | 0.33 | 0.33 |
| **query "pseudo-example": positives** | **62** | **97** | **136** | **179** |
| query "pseudo-example": F1 | 0.20 | 0.26 | 0.30 | **0.38** |
| LTS-lite: positives (± sd) | 0.4 | 1.3 | 5.7 ± 9.9 | 26.2 ± 21.1 |
| LTS-lite: F1 | 0.03 | 0.08 | 0.12 | 0.30 |
| **ceiling** (all 143K pool labels) | | | | F1 **0.43**, AP 0.38 |

### identity_hate (pool 0.88%, 1,257 positives; test 712 positives)
| Sampler | B=100 | 250 | 500 | 1000 |
|---|---|---|---|---|
| random: positives / F1 | 1.1 / 0.11 | 2.2 / 0.18 | 4.3 / 0.26 | 10.6 / 0.38 |
| query "description": positives / F1 | **0 / 0.00** | 2 / 0.03 | 8 / 0.21 | 22 / 0.32 |
| query "pseudo-example": positives / F1 | 24 / 0.21 | 58 / 0.34 | 99 / **0.39** | 162 / 0.36 |
| LTS-lite: positives / F1 | 1.7 / 0.10 | 9.4 / 0.26 | 40.7 / 0.36 | 118.3 / **0.42** |
| **ceiling** | | | | F1 **0.46**, AP 0.49 |

## Findings
1. **Random fails, as expected.** For threat at B ≤ 500, a classifier can only be trained in 20–70% of runs, and mean F1 is ≤ 0.09. This is the motivating failure.
2. **How the query is phrased matters enormously.** Writing it like a comment (*"I am going to find you and kill you."*) finds **22x more threats** than a description (179 vs 8 at B = 1000). For identity_hate, the description at B = 100 found **zero** positives, worse than random. This confirms the model-card concern: MiniLM is a *symmetric* similarity model, so a predicate description doesn't look like a comment. **Implications:** (a) query-seeded results depend on human wording, a confound to report; (b) it strengthens the case for retrieval-tuned (asymmetric) encoders as a second arm; (c) phrasings could themselves be bandit arms.
3. **One query already gets 42% of all pool threats in 1,000 labels** (179/427), and its proxy (F1 0.38, AP 0.39) is close to the full-data ceiling (AP 0.38). Smart sampling can recover most of the available performance.
4. **The ceiling is low.** LR on MiniLM features tops out around F1 0.43 (threat) and 0.46 (identity_hate), even with every label. The **proxy representation**, not the sample, limits us. Before comparing samplers, decide whether the fixed proxy-feature embedding should be stronger (mpnet / bge). Samplers can only be compared inside the gap between floor and ceiling, and it's currently narrow.
5. **More positives ≠ proportionally better F1.** Pseudo-example threat goes from 62 to 179 positives while F1 only goes from 0.20 to 0.38. One possible reason is that one query finds one *kind* of threat (low diversity). Another is that the threshold comes from only 47 val positives (threat), so it's noisy. Both point to diversity metrics and CIs.
6. **Few hard examples can beat random:** "description" threat finds only 3–8 positives yet beats random at every budget, probably because its negatives are *hard* (toxic but not threatening) and teach the boundary better.

7. **LTS-lite (competing baseline) has a cold-start problem, then catches up.** At B ≤ 250 it equals random: with no positives, Thompson sampling explores clusters blindly. Once a batch hits a positive-rich cluster it exploits it, so the positive count jumps at B = 1000 (threat 26, identity_hate 118), with **very high variance across seeds** (sd ≈ the mean). Positives *are* concentrated: the top 5 of 50 k-means clusters hold 72% (threat) and 68% (identity_hate) of positives, so cluster-level exploitation is viable.
8. **LTS-lite has the best identity_hate F1 at B = 1000 (0.42), despite fewer positives than the pseudo-example query (118 vs 162).** It draws from several clusters, while one query draws from one neighborhood. That's the first evidence for the diversity argument (finding 5): *which* positives matter, not just how many. Single runs, so not yet significant.

**Implication for our method:** the bandit arms need a warm start. A purely blind bandit wastes the first rounds. Seeding each embedding arm with query-based retrieval (finding 2) should fix LTS-lite's cold start; LTS's own paper doesn't address it.

## Caveats
- Query runs are deterministic, one run each, with no variance estimate. Random runs have 10 seeds (std 0.07–0.15 F1).
- **Validation positives are few** (threat 47, identity_hate 140), so threshold choice is noisy. Average precision (threshold-free) is the more stable comparison.
- LTS-lite deviates from the paper: the reward is "batch found a positive" (not validation-F1 gain), LR instead of fine-tuned BERT, batch 25 (not 200), k-means k = 50 (not topic modeling). See `lts_lite` docstring.
- One phrasing per variant, picked by hand, not tuned. The results are sensitive to wording (finding 2).
- Exact-text dedup only; near-duplicates are not handled yet.

## Next steps
- [ ] **Decide the proxy reference embedding** (finding 4): embed with `all-mpnet-base-v2` and `bge-base-en-v1.5`, rerun the ceiling for each.
- [ ] Positive-anchor sampler (d): seed with query results, expand via kNN of found positives (SEALS/AnchorAL).
- [ ] RQ1 diagnostics: do MiniLM / mpnet / bge retrieve *different* threats? (top-k overlap, connected components of positives)
- [ ] Bootstrap CIs on test F1/AP for the deterministic samplers.
