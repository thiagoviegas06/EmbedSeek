# [1] Chung et al. 2026: 100x Cost & Latency Reduction with Lightweight Proxy Models

**Citation:** Y. Chung, R. Desai, J. He, … A. Halevy, Y. Papakonstantinou (Google Cloud). *Proc. ACM Manag. Data* 4(3), SIGMOD 2026. doi:10.1145/3802002. arXiv:2603.15970.
**PDF:** [papers/chung2026_proxy_models.pdf](../../papers/chung2026_proxy_models.pdf)
**Role in our project:** the **baseline pipeline** we build on. Our project starts where this paper's sampling step fails.

## One-line summary
Many semantic operators (`AI.IF`, `AI.RANK`) are really classification. So label ~200–1,000 rows with an LLM, train **logistic regression on text embeddings**, and run that on the whole table. You get >100x cost and latency savings at roughly LLM-level F1, and fall back to the LLM when the proxy isn't good enough.

## The pipeline (Fig. 1)
1. Parse the query into operator `O`, prompt `Q`, column `C`.
2. **Embed** the column (Gecko `text-embedding-005`, 768-d; precomputed if available).
3. **Sample** rows. The default is **random**.
4. **LLM-label** the sample (Gemini 2.5 Flash).
5. **Train LR**: sklearn defaults except `class_weight="balanced"`. No hyperparameter tuning.
6. **Adaptive proxy selection**: deploy the proxy only if `|τ(proxy) − τ(LLM)| ≤ t` (e.g. t = 0.1). Otherwise fall back to the LLM. Accuracy is checked against LLM labels on the training sample.
- *Online* (BigQuery, ad hoc): sample, label, and train inside the query.
- *Offline* (AlloyDB): pre-train for known queries, which gives sub-second latency.

## Key results
- 10M-row Amazon Polarity: online proxy is **329x faster / 728x cheaper**; offline is **991x / 792x** (Table 1).
- Relative macro F1 (proxy / LLM) is ~0.9–1.16 across 11 classification datasets (Table 5). The proxy sometimes *beats* the LLM because LR acts as a regularizer.
- **Toxic Conversations** (imbalance 11.6:1): proxy F1 0.653 vs LLM 0.648. This is the closest benchmark to our Jigsaw setting.
- Cost is dominated by **LLM labeling**, which is fixed regardless of table size, so savings grow with table size. With on-the-fly embedding (not cached), savings drop to only ~2.5x cost and 5x latency, because embedding costs about a third as much as LLM inference (Table 6). → **Embedding cost matters a lot for us**, since we use *multiple* encoders.
- LR vs SVM / RF / XGB: tuning brings no gain, and LR is fastest (Table 13). This justifies our choice of a fixed LR proxy.

## Sections most relevant to us
### §5.4 Sampling strategy, and the failure we target
- They compare **Top-K similarity**, **Random**, and **AL (stratified)**: AL uses proxy confidence and always grabs predicted-minority rows.
- At high imbalance (Toxic Conversations, ρ = 11.6), **both Top-K and Random fail** to build a balanced sample (Fig. 4a).
- **IR / needle settings (SciFact, 1.1 relevant docs per 5K) are "near impossible"**: the imbalance ratio grows linearly with samples (Fig. 4c). The proxy fails on SciFact, FiQA, and SCIDOCS (Table 8: SciFact proxy nDCG@10 0.0098 vs LLM 0.508).
- Top-K "blindly favors one class" by query relevance and gives *higher* imbalance than random in some cases.
- **Sampling overhead** (Table 10, 52K rows, 1,000 samples, serial): Top-K is **43,761x** and AL is **51,639x** the latency of random. → Our cost accounting must take the sampler's overhead seriously.

### §5.5 Imbalanced training
- Balanced class weights are the default. When minority examples are scarce (≤100), use SMOTE / oversampling. At imbalance ≥50, downsampling does best.
- Training is ≤2% of end-to-end time.

### §5.6 Embedding quality
- Gecko (768-d) ≥ Gemini (3072-d) on Tweet Sentiment. Bigger embeddings aren't always better.
- Gemma-300m (the open, cheap model) is consistently worst.
- They use PCA plus a "separability score" (inter-class distance / intra-class variance) to explain the differences.
- → This is direct support for our **RQ1**: the choice of embedding changes proxy quality. But they only vary the *classifier's* embedding, never the *sampler's*.

### §6.2 Open problem stated explicitly
> "A significant challenge remains the 'Needle-in-a-Haystack' sampling problem; obtaining sufficient representative examples for every class-label is essential…"

## Gaps our project fills
| Chung et al. | EmbedSeek |
|---|---|
| One embedding (Gecko) for both sampling and the proxy | Several embeddings, with the *sampler* choosing among them |
| Random / Top-K / AL sampling, evaluated by imbalance ratio only | Query, anchor, diversity, and bandit samplers; evaluated by positive yield, **diversity**, and downstream F1 |
| Says the needle case is "near impossible" and falls back to the LLM | Tries to make the needle case work |
| LLM labels as ground truth | Human gold labels for the test set |

## Things to borrow
- LR with `class_weight="balanced"` and default hyperparameters as the proxy. This matches our §3.5.
- The cost breakdown (embedding / sampling / labeling / training / inference) and the cached vs on-the-fly distinction. This is our `C_adaptive`.
- The Toxic Conversations dataset (MTEB `toxic_conversations_50k`) is a candidate MTEB dataset that is directly comparable to their numbers.
- The separability score as one more RQ1 diagnostic, which is cheaper than UMAP.

## Questions / caveats
- Their "accuracy" is relative to LLM labels. Proxy F1 is capped by LLM quality ("when LLM accuracy is low, so is the proxy").
- They give no confidence intervals or seeds. We should do better (10 seeds).
- The latency multipliers depend on their infrastructure and parallelism, so we shouldn't compare our absolute numbers to theirs.
