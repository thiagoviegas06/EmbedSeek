# [5] Coleman et al. 2022: SEALS (Similarity Search for Efficient Active Learning and Search of Rare Concepts)

**Citation:** C. Coleman, E. Chou, J. Katz-Samuels, S. Culatana, P. Bailis, A. C. Berg, R. Nowak, R. Sumbaly, M. Zaharia, I. Z. Yalniz (Stanford / Facebook AI / UW). AAAI 2022. arXiv:2007.00077.
**PDF:** [papers/coleman2022_seals.pdf](../../papers/coleman2022_seals.pdf)
**Role in our project:** the basis for our **positive-anchor sampler (d)**. It also gives us the **diagnostics that predict when neighbor search works**, which are our RQ1 metrics.

## One-line summary
Don't scan the whole unlabeled pool each active learning round. Restrict candidates to the **k nearest neighbors of already-labeled examples** and grow that pool as you label. You get the same mAP and recall as a global scan at up to 1000x less selection compute, because rare concepts tend to cluster in pretrained embedding space.

## Algorithm (Alg. 2)
```
P = ∪_{labeled z} kNN(z, k)                # candidate pool = neighbors of the labeled set
repeat:
    A = train(L)                           # logistic regression on embeddings
    for 1..b:
        z* = argmax_{z∈P} φ(z, A, P)       # selection strategy applied to the pool only
        L = L ∪ {(z*, label(z*))}
        P = (P \ {z*}) ∪ kNN(z*, k)        # expand with the new point's neighbors
until |L| = T
```
- **Seed:** 5 positives + 95 random negatives. Batch b = 100, budget T = 2,000, k = 100 (1K–10K for the 10B-image set).
- **Selection strategies φ:**
  - **MaxEnt** (uncertainty; for binary this equals least-confidence and margin)
  - **ID** (information density = uncertainty × mean similarity to the pool; O(n²) globally)
  - **MLP** (most-likely positive = P(positive)). This is **the active *search* strategy**: it maximizes positives found.
- Index: FAISS LSH (Euclidean); embeddings l2-normalized.
- **Greedy k-centers (coreset) never beat random** in their setting. Pure coverage ignores labels, and rare concepts are a tiny fraction of the space. → **Warning for our diversity sampler (c):** pure coverage alone will likely fail at 0.1%.

## Key results
| Dataset | Concepts | Prevalence | Finding |
|---|---|---|---|
| ImageNet (640K, unseen 500 classes) | 450 | 0.11–0.20% | All SEALS strategies within 0.001 mAP of global; random gets 0.436 vs ~0.69. **MLP recall 74%** after labeling 0.31% of the data. |
| OpenImages (6.8M) | 153 | 0.002–0.088% | SEALS 0.386 vs global 0.399 mAP; larger k closes the gap (0.395 at k = 1000). Global ID didn't finish in 24h. |
| 10B images | 8 | — | Selection in seconds on 1 machine vs a cluster. **MLP-SEALS did poorly at this scale: near-duplicates.** Still 100x more positives than random. |
| **Goodreads spoilers (text, SBERT-NLI-base, PCA → 256d)** | 1 | 3.2% | Active search: SEALS ≈ global, up to 10x random. **Active learning: no strategy beat random**, see below. |

- Random-SEALS (random choice from the kNN pool) gets 0.498 mAP vs 0.436 for Random-All. So most of the gain comes from the *selection strategy*, not just the reduced pool skew.
- Embedding ablation (App. 7.4): ResNet-18/50 at 128/256/512-d gave similar SEALS results (images only).

## When it works: latent structure (§6, Fig. 3)
- AP correlates with the **size of the largest connected component** of positives in the kNN graph (ρ ≈ 0.6) and with a **shorter average shortest path** between positives (ρ ≈ −0.7).
- Specific concepts ("monster truck", "blackberry") cluster tightly and work. Generic ones ("electric blue", "meal") are scattered and work worse.
- **Goodreads text failure (App. 7.12):** with generic Sentence-BERT, spoiler sentences had an **average pairwise cosine similarity of 0.004**, i.e. nearly uniform across the space (ImageNet concepts: 0.45; OpenImages: 0.36). Spoilers are book-specific, but SBERT organizes by topic, so positives scatter across topics. The authors call this "a limitation and opportunity for future work."

## Why this matters for EmbedSeek
1. **Direct motivation for multiple embeddings (RQ1):** SEALS's one text experiment failed because the *representation* didn't put positives together. A different encoder might. That's our hypothesis.
2. **RQ1 diagnostics to borrow** (computed per encoder, per predicate, *retrospectively* with gold labels):
   - size of the largest connected component of positives in the kNN graph (at k = 10 and 100)
   - average shortest path between positives
   - mean pairwise cosine among positives vs among random pairs
   These *predict* anchor-search success, so they're better than UMAP pictures.
3. **Sampler (d) positive-anchor** = SEALS with MLP selection, in each embedding. A simpler version: rank the kNN pool by similarity to the anchors and skip LR retraining.
4. **Threat ≈ "harmful content"**, the motivating example in their intro. But Jigsaw threats co-occur with general toxicity (94%), so anchor neighborhoods may fill with non-threat toxic comments. We expect a mid-way case between ImageNet and Goodreads.
5. **Near-duplicates:** their 10B failure mode. Our Jigsaw open item already flags dedup.

## Gaps they leave (that we address)
- **One fixed embedding per dataset.** No per-predicate choice among encoders.
- The seed requires 5 known positives. In our setting a query-seeded first round has to *find* them.
- No diversity measurement of the positives found (only count and recall).
- Images mostly; only one text appendix experiment.

## Questions / caveats
- With 5 seed positives, a pool limited to neighbors can get **trapped in one positive component**. This is why our proposal adds an *exploration fraction* to sampler (d). AnchorAL addresses this too (see [04_lesci2024_anchoral.md](04_lesci2024_anchoral.md)).
- Their 5 runs report a mean and standard deviation. Matching or exceeding this (10 seeds) is reasonable.
