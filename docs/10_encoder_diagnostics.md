# Encoder diagnostics: go/no-go for the encoder layer (2026-11-14)

**Question:** do MiniLM, mpnet and bge concentrate *different* positives? If they all hold the same ones, the encoder level of the two-level bandit ([07_updated_direction.md](07_updated_direction.md)) has nothing to choose between.

**Verdict: a qualified go.** The encoders are complementary, but less at the cluster level (where the bandit works) than at the query level, and bge is the best single encoder on almost every measure. The decision is the humans' to confirm; see [Verdict](#verdict).

**Setup:** frozen Jigsaw splits, GPU embeddings from job 22865730 (each model's own max length). Slurm job 23156866, 3 min on 8 CPUs.
**Reproduce:** `sbatch scripts/hpc/cpu.sbatch scripts/encoder_diagnostics.py data/jigsaw` → `results/encoder_diagnostics.csv` (log: `logs/embedseek-cpu-23156866.out`).

This is a retrospective analysis: it ranks clusters with the true pool labels. Nothing here may be fed into a sampler.

## 1. Ceilings: the proxy trained on all 143K pool labels

Test F1 and AP with 95% test-set bootstrap CIs.

| Encoder | threat F1 | threat AP | identity_hate F1 | identity_hate AP |
|---|---|---|---|---|
| MiniLM | 0.43 [0.37, 0.49] | 0.38 [0.32, 0.45] | 0.46 [0.43, 0.49] | 0.49 [0.45, 0.52] |
| mpnet | 0.43 [0.37, 0.48] | 0.42 [0.35, 0.49] | 0.46 [0.43, 0.49] | 0.47 [0.43, 0.51] |
| **bge** | **0.48** [0.42, 0.52] | **0.44** [0.37, 0.51] | **0.52** [0.49, 0.54] | **0.55** [0.51, 0.59] |

- bge has the highest ceiling on both predicates and both metrics: about +0.05 F1 over the other two.
- mpnet is no better than MiniLM on F1, despite twice the dimension and 5x the embedding time.
- The threat CIs overlap heavily. For identity_hate, bge's interval sits just above the others. These are separate CIs, not a paired bootstrap, so the gaps are not yet tested.
- The ceiling is still low (F1 ≈ 0.5). A stronger encoder raises it a little; it does not remove the limit noted in [06_baseline_results.md](06_baseline_results.md), finding 4.

## 2. Clusters: k-means (k = 50), top-5 clusters by positive count

| Encoder | threat: share of positives | threat: share of pool rows | identity_hate: share of positives | identity_hate: share of pool rows |
|---|---|---|---|---|
| MiniLM | 72% | 12.9% | 71% | 10.5% |
| mpnet | 75% | 12.4% | 69% | 11.7% |
| **bge** | **78%** | 10.5% | **79%** | 11.6% |

Every encoder packs 69–79% of positives into clusters covering 10–13% of the pool, so cluster-level exploitation is viable in all three. bge concentrates the most.

**Do the top clusters hold the same positives?** Mostly yes.

| | threat (427 positives) | identity_hate (1,257 positives) |
|---|---|---|
| Pairwise Jaccard of held positives | 0.70 – 0.78 | 0.69 – 0.77 |
| Held by all three encoders | 243 (57%) | 695 (55%) |
| Held by the best single encoder (bge) | 335 (78%) | 989 (79%) |
| Held by at least one encoder | 390 (91%) | 1,140 (91%) |
| Held only by MiniLM / mpnet / bge | 12 / 12 / 39 | 41 / 38 / 147 |

- Using all three encoders raises reachable positives from 78–79% to 91%: +55 threats and +151 identity_hate positives over bge alone.
- Most of the unique positives are bge's. MiniLM and mpnet overlap each other the most (Jaccard 0.78 and 0.77) and each adds little on its own.

## 3. Query: top-1000 pool rows for the comment-like query

| Encoder | threat positives | identity_hate positives |
|---|---|---|
| MiniLM | 179 | 162 |
| mpnet | 196 | 174 |
| bge, with instruction prefix | **221** | 169 |
| bge, no prefix | 218 | **191** |

| | threat | identity_hate |
|---|---|---|
| Jaccard of top-1000 rows, between encoders | 0.25 – 0.29 | 0.22 – 0.26 |
| Jaccard of positives found, between encoders | 0.62 – 0.65 | 0.36 – 0.41 |
| Positives found by the best single encoder | 221 | 174 |
| Positives found by at least one encoder (prefixed bge) | 266 | 283 |

- The encoders retrieve largely different rows for the same query (Jaccard ≈ 0.25).
- For threat, they still land on mostly the same positives (0.62–0.65). For identity_hate they do not (0.36–0.41): the union finds 283 positives against 174 for the best single encoder.
- The bge instruction prefix makes no difference for threat (221 vs 218) and hurts for identity_hate (169 vs 191). That matches the model card: no instruction for symmetric, comment-to-comment similarity. The two bge variants also disagree with each other more than expected (top-1000 Jaccard 0.55 and 0.43).

## Verdict

HPC.md's criteria: **go** if encoders concentrate meaningfully different positives, or if different encoders win on different predicates; **no-go** if overlap is high everywhere.

| Criterion | Result |
|---|---|
| Different encoders win on different predicates | **No.** bge wins both ceilings and both cluster concentrations. Only the query ranking changes order, and only between bge variants and mpnet |
| Different positives at the cluster level | **Weakly.** Jaccard 0.7–0.78, but the union reaches 91% of positives against 78–79% for bge alone |
| Different positives at the query level | **Yes for identity_hate** (Jaccard ≈ 0.4, union +63%), **modestly for threat** (Jaccard ≈ 0.63, union +20%) |

**Qualified go.** Overlap is not high everywhere, so this is not a no-go. But the case is weaker than the proposal assumed, in two ways:

1. **The bar to beat is LTS-lite on bge alone.** bge is the best fixed encoder almost everywhere, so the encoder level mostly has to *discover that online* rather than exploit a predicate-dependent winner. Its remaining upside is the 12–13 points of positives that only another encoder's clusters hold.
2. **Complementarity is strongest in query retrieval, not in clusters.** This supports the `warm_start` option (one query-seeded batch per encoder) and suggests query search within an encoder may matter more than docs/07 assumed.

**The union numbers overstate the gain.** The union of three encoders' top-5 clusters spans up to 15 clusters, and the query union spends up to 3,000 labels against 1,000. Neither is an equal-budget comparison. The fair versions (bge's top-15 clusters against the three-encoder union; bge's top-3000 against 3 × top-1000) were not run. They are a small addition to the script and would tighten this verdict before the bandit is built.

## Proposal for task 4: the fixed proxy embedding

**Use `bge-base-en-v1.5`.** It has the highest ceiling on both predicates and both metrics, at the same embedding cost as mpnet (220 s vs 225 s for the pool).

Open points for the humans (HPC.md §6):
- If bge is both the proxy embedding and a search arm, positives found in bge's space may suit the proxy better than those found elsewhere. That would favor the bge arm for a reason unrelated to search quality.
- All existing baselines use MiniLM as the proxy. Switching means re-running them on bge features.
- The ceiling gap over MiniLM is not yet confirmed with a paired bootstrap.

## Caveats

- **One clustering.** k-means used a single seed (`random_state=0`, `n_init=1`). LTS-lite already proved sensitive to the clustering (HPC.md §4), so the cluster shares and Jaccards may move with another seed.
- **Two predicates, one dataset, one query phrasing each.** Query results are known to depend heavily on wording (docs/06, finding 2).
- **Top-5 is by true positive count.** A sampler would not know which clusters these are; the shares are an upper bound on what cluster exploitation can reach.
- MiniLM's identity_hate share is 71% here against 68% in docs/06: the GPU embeddings give a different clustering from the laptop run.
