# Project overview

## The problem
A *semantic filter* is a query like `AI.IF('contains a credible threat', comment)` run over a big table. Calling an LLM on every row is slow and expensive. The known fix (Chung et al. [1]): label a small sample with the LLM, train a cheap **proxy model** (logistic regression on text embeddings), and run the proxy on the rest.

**Failure case:** when positives are rare, a uniform sample has almost none. At 0.1% prevalence, 500 random labels give ~0.5 positives, so the proxy learns nothing. Reweighting can't help for positives never seen.

## Our angle ("Direction B")
Find diverse positives cheaply by sampling smartly. Prior work (SEALS, AnchorAL, Barbosa et al.) searches neighbors in *one* embedding space. We ask: does the **choice of embedding** matter, and can a policy **learn which embedding is most productive per predicate** from a little LLM feedback?

## Research questions
- **RQ1 Representation:** how much do rare-positive neighborhoods and rankings differ across embeddings?
- **RQ2 Discovery:** do query-similarity / positive-anchor / diversity samplers find more (and more varied) positives per label than random?
- **RQ3 Adaptation:** does a bandit that picks among embeddings beat the *best single fixed* embedding at the same budget?
- **RQ4 Downstream + cost:** do better samples give better held-out proxy F1, and what does the extra compute cost?

## Method pieces
1. **Embeddings:** 2–3 frozen pretrained encoders from different families (e.g. all-MiniLM-L6-v2 general; a retrieval-tuned one like bge/e5). No fine-tuning. Normalize vectors; compare via within-model ranks, never raw cross-model coordinates.
2. **Fixed-representation samplers**, budgets B ∈ {100, 250, 500, 1000}:
   (a) uniform random (baseline) (b) query-seeded retrieval (cosine to predicate text) (c) diversity / cluster coverage (d) positive-anchor neighbor search with an exploration fraction.
   **Competing baseline:** LTS-lite, a simplified version of Barbosa et al.: Thompson sampling over k-means clusters of a single embedding. It isolates whether gains come from *having a bandit* or from *bandit arms = embedding spaces*.
3. **Adaptive method (core contribution):** each embedding = a bandit arm. Per round: pick arm, propose a batch, get labels, update. Thompson sampling or UCB. Reward `r = (p + λ·d) / b` (new positives, novel positive types, labels used).
4. **Proxy:** same class-weighted logistic regression, fixed reference embedding, threshold picked on validation. Same for every sampler so only *selection* varies.
5. **Labels:** first simulate with existing human labels as a hidden oracle (cheap, repeatable, 10 seeds); then a bounded real-LLM run to measure label noise.

## Optional / out of scope
Optional: learned weights to combine embedding similarities. Out of scope: fine-tuning encoders, a production DB engine.

## What counts as success
A well-controlled answer to *when* adaptive representation choice helps. A carefully documented negative result is fine.
