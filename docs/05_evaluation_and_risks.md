# Evaluation and risks

## Acquisition metrics
- Positives found per 100 labels; cumulative positives vs budget
- Positive-subtype coverage, measured **independently** of the sampler's own embedding (use dataset subtype labels, or one shared fixed embedding + manual inspection)
- Bandit: reward and arm allocation per round

## Downstream (identical protocol for all samplers)
Class-weighted logistic regression on one fixed reference embedding; threshold chosen on validation; report F1, precision, recall, average precision on the held-out human-labeled test set.
**Key comparison:** adaptive vs the *best fixed single-embedding* sampler, not just vs random.

## Cost
`C_adaptive = Σ_m C_embed,m + C_index + C_sample + B·c_LLM + C_train + C_infer`
Report cold-start (embed everything) vs cached embeddings. Don't assume savings; show them.

## Pitfalls to watch
- 2D UMAP/PCA separation ≠ classifier separability.
- Leaking test labels into sampler development; splits must stay disjoint.
- Self-referential diversity (clusters from the sampler's own embedding).
- Too few positives in val/test at 0.1% prevalence → thresholds and F1 unstable; use CIs.
- Noisy labels (see Jigsaw `threat` note in [04_jigsaw_exploration.md](04_jigsaw_exploration.md)).
- Cross-model similarity scales differ: compare ranks.
